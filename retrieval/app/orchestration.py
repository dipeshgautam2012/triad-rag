"""Wire chunkers and indexers — handles, metadata, ingest, search."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from llama_index.core.schema import BaseNode

from app.chunkers.chunker_factory import CHUNKERS, make_chunker
from app.chunkers.hierarchical_chunker import HierarchicalChunker
from app.config import settings
from app.embedders.embedder_factory import make_embedder
from app.embedders.base_embedder import BaseEmbedder
from app.hybrid import combine_hybrid_results, format_retrieved, node_from_retrieved
from app.indexers.base_indexer import validate_index_id as check_index_id
from app.indexers.bm25_indexer import Bm25Indexer
from app.indexers.chroma_indexer import ChromaIndexer
from app.indexers.indexer_factory import make_bm25_indexer, make_chroma_indexer
from app.ingest.corpus import corpus_dir, list_corpus_files, unlink_corpus_file
from app.ingest.filename_sanitizer import sanitize_corpus_filename
from app.ingest.upload import save_upload
from app.rerankers.base_reranker import BaseReranker
from app.rerankers.reranker_factory import make_reranker
from app.stores.base_vector_store import IndexSnapshotError
from app.stores.chroma_vector_store import list_indices_detailed, write_index_description
from app.stores.store_factory import make_node_store, make_sparse_store, make_vector_store

INDEXER_MODES = ("chroma", "bm25", "hybrid")


@dataclass
class IndexHandles:
    chroma: ChromaIndexer
    bm25: Bm25Indexer


def store_root() -> Path:
    return Path(settings.index_store_dir).resolve()


def corpus_root() -> Path:
    return Path(settings.corpus_dir).resolve()


def validate_index_id(index_id: str) -> None:
    try:
        check_index_id(index_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _vector_store(index_id: str):
    return make_vector_store(index_id, backend=settings.vector_backend, store_root=store_root())


def _normalize_mode(mode: str | None) -> str | None:
    val = (mode or "").strip().lower()
    if not val:
        return None
    return "chroma" if val == "vector" else val


def available_indexers() -> list[str]:
    if settings.sparse_backend.strip().lower() == "none":
        return ["chroma"]
    return list(INDEXER_MODES)


def read_indexer_mode(index_id: str) -> str | None:
    col = _vector_store(index_id).try_get_collection()
    if col is None:
        return None
    return _normalize_mode(str((col.metadata or {}).get("indexer", "")))


def write_indexer_mode(index_id: str, mode: str) -> None:
    store = _vector_store(index_id)
    md = dict(store.get_collection().metadata or {})
    if md.get("indexer") != mode:
        md["indexer"] = mode
        store.modify_metadata(md)


def resolve_ingest_mode(index_id: str, requested: str | None) -> str:
    available = available_indexers()
    req = _normalize_mode(requested)
    if req and req not in available:
        raise HTTPException(status_code=400, detail=f"indexer must be one of {available}")
    recorded = read_indexer_mode(index_id)
    if recorded and req and req != recorded:
        raise HTTPException(
            status_code=409,
            detail=f"index {index_id!r} uses {recorded!r}; cannot ingest with {req!r}",
        )
    mode = req or recorded or "chroma"
    if mode not in available:
        raise HTTPException(
            status_code=400,
            detail=f"index {index_id!r} uses {mode!r} but sparse_backend is 'none'",
        )
    return mode


def _ensure_collection(
    index_id: str,
    *,
    embedding_model: str,
    chunker: str,
    description: str | None,
) -> None:
    """Ensure collection exists and is configured correctly."""
    store = _vector_store(index_id)
    if store.try_get_collection() is None:
        store.create_collection(
            embedding_model=embedding_model,
            chunker=chunker,
            description=(description or "").strip()[:500] or None,
        )
        return
    if embedding_model != store.resolve_embedding_model():
        raise IndexSnapshotError(
            f"index {index_id!r} uses embedding_model {store.resolve_embedding_model()!r}; "
            f"cannot ingest with {embedding_model!r}"
        )
    if chunker != store.resolve_chunker():
        raise IndexSnapshotError(
            f"index {index_id!r} uses chunker {store.resolve_chunker()!r}; "
            f"cannot ingest with {chunker!r}"
        )


def index_handles(
    app: FastAPI,
    index_id: str,
    *,
    cache: bool = True,
    chroma_embedder: BaseEmbedder | None = None,
) -> IndexHandles:
    validate_index_id(index_id)
    if cache and index_id in app.state.indices:
        h = app.state.indices[index_id]
        if chroma_embedder is not None:
            h.chroma.bind_embedder(chroma_embedder)
        return h
    root = store_root()
    node = make_node_store(index_id, backend=settings.node_store_backend, store_root=root)
    h = IndexHandles(
        chroma=make_chroma_indexer(
            index_id,
            embedding_store=make_vector_store(
                index_id, backend=settings.vector_backend, store_root=root
            ),
            lookup_store=node,
            embedder=chroma_embedder,
        ),
        bm25=make_bm25_indexer(
            index_id,
            keyword_store=make_sparse_store(
                index_id, backend=settings.sparse_backend, store_root=root
            ),
            context_store=node,
        ),
    )
    if cache:
        app.state.indices[index_id] = h
    return h


def _ensure_chroma_embedder(chroma: ChromaIndexer) -> None:
    """Bind an embedder when chroma.embedding_model is set (e.g. after load())."""
    if chroma.embedding_model:
        chroma.bind_embedder(
            make_embedder(chroma.embedding_model, backend=settings.embedder_backend)
        )


def ensure_loaded(h: IndexHandles) -> None:
    if h.chroma.ready or h.bm25.ready:
        return
    try:
        h.chroma.load()
        h.bm25.load()
        _ensure_chroma_embedder(h.chroma)
    except IndexSnapshotError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


def startup(app: FastAPI) -> None:
    unknown = {c.strip().lower() for c in settings.available_chunkers} - CHUNKERS
    if unknown:
        raise ValueError(f"unimplemented chunkers: {sorted(unknown)}")
    failures = []
    for model in settings.available_embedding_models:
        name = str(model).strip()
        if not name:
            continue
        try:
            make_embedder(name, backend=settings.embedder_backend)
        except Exception as e:
            failures.append(f"{name!r}: {e}")
        else:
            logging.info("Preloaded embedding model %s", name)
    if failures:
        raise ValueError("embedding models failed to load: " + "; ".join(failures))
    corpus_root().mkdir(parents=True, exist_ok=True)
    store_root().mkdir(parents=True, exist_ok=True)
    app.state.indices = {}
    app.state.reranker = make_reranker(settings.rerank_model, backend=settings.reranker_backend)
    h = index_handles(app, "default")
    try:
        h.chroma.load()
        h.bm25.load()
        _ensure_chroma_embedder(h.chroma)
    except IndexSnapshotError as e:
        logging.error("Default index load failed: %s", e)
        raise


def search_index(
    h: IndexHandles,
    *,
    mode: str,
    query: str,
    top_k: int,
    rerank: bool | None,
    reranker: BaseReranker,
    expand: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    k = max(1, int(top_k))
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    use_expand = settings.search_expand if expand is None else expand
    q = query.strip()
    resolved = mode.strip().lower()
    sparse_on = settings.sparse_backend.strip().lower() != "none"

    if resolved == "bm25":
        if not h.bm25.ready:
            return [], []
        retrieve_k = max(k, k * settings.rerank_candidate_multiplier) if use_rerank else k
        raw = h.bm25.search(q, retrieve_k, expand=use_expand)
        if not use_rerank:
            return raw[:k], []
        hits = reranker.rerank([node_from_retrieved(x) for x in raw], q, top_n=k)
        return [format_retrieved(x) for x in hits], raw

    if not h.chroma.ready:
        return [], []

    if resolved == "hybrid" and sparse_on:
        retrieve_k = k * settings.hybrid_candidate_multiplier
        if use_rerank:
            retrieve_k = max(retrieve_k, k * settings.rerank_candidate_multiplier)
        vector_hits = [
            node_from_retrieved(x)
            for x in h.chroma.search(q, retrieve_k, expand=use_expand)
        ]
        sparse_hits = h.bm25.search(q, retrieve_k, expand=use_expand)
        hits = combine_hybrid_results(vector_hits, sparse_hits, limit=retrieve_k)
        pool = [format_retrieved(x) for x in hits]
        if use_rerank:
            hits = reranker.rerank(hits, q, top_n=k)
            return [format_retrieved(x) for x in hits], pool
        return pool[:k], pool

    retrieve_k = max(k, k * settings.rerank_candidate_multiplier) if use_rerank else k
    raw = h.chroma.search(q, retrieve_k, expand=use_expand)
    if use_rerank:
        hits = reranker.rerank([node_from_retrieved(x) for x in raw], q, top_n=k)
        return [format_retrieved(x) for x in hits], raw
    return raw, []


def ingest_file(
    app: FastAPI,
    *,
    index_id: str,
    data: bytes,
    filename: str | None,
    index_description: str | None,
    embedding_model: str | None,
    chunker_name: str | None,
    indexer: str | None,
) -> dict[str, Any]:
    if embedding_model and embedding_model not in settings.available_embedding_models:
        raise HTTPException(
            status_code=400,
            detail=f"embedding_model must be one of {settings.available_embedding_models}",
        )
    if chunker_name and chunker_name not in settings.available_chunkers:
        raise HTTPException(
            status_code=400,
            detail=f"chunker_name must be one of {settings.available_chunkers}",
        )

    mode = resolve_ingest_mode(index_id, indexer)
    emb_name = (embedding_model or "").strip() or settings.default_embedding_model
    chunker_key = (chunker_name or "").strip() or settings.chunker_name
    ingest_embedder = (
        make_embedder(emb_name, backend=settings.embedder_backend)
        if mode in ("chroma", "hybrid")
        else None
    )
    h = index_handles(app, index_id, chroma_embedder=ingest_embedder)
    try:
        saved = save_upload(
            data, filename, corpus_dir(corpus_root(), index_id), max_bytes=settings.max_upload_bytes
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    embedder = ingest_embedder or make_embedder(emb_name, backend=settings.embedder_backend)
    chunker = make_chunker(
        chunker_key,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embed_model=embedder.embedding_model if chunker_key == "semantic" else None,
        hierarchical_parent_multiplier=settings.hierarchical_parent_multiplier,
        hierarchical_chunk_sizes=settings.hierarchical_chunk_sizes,
        hierarchical_embed_at=settings.hierarchical_embed_at,
        sentence_window_size=settings.sentence_window_size,
        semantic_breakpoint_percentile=settings.semantic_breakpoint_percentile,
        semantic_buffer_size=settings.semantic_buffer_size,
    )
    result = chunker.chunk_file(saved)
    lookup_nodes: list[BaseNode] | None = (
        chunker.hierarchy_nodes if isinstance(chunker, HierarchicalChunker) else None
    )

    try:
        _ensure_collection(
            index_id, embedding_model=emb_name, chunker=chunker_key, description=index_description
        )
        if mode in ("chroma", "hybrid"):
            h.chroma.remove_source(saved.name)
            h.chroma.add_chunks(result, lookup_nodes=lookup_nodes)
            h.chroma.load()
        if mode in ("bm25", "hybrid"):
            h.bm25.remove_source(saved.name)
            h.bm25.add_chunks(result, write_context=mode != "hybrid")
    except IndexSnapshotError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    write_indexer_mode(index_id, mode)

    bm25_only = mode == "bm25"
    return {
        "index_id": index_id,
        "saved_as": saved.name,
        "chunks_indexed": h.bm25.chunk_count() if bm25_only else h.chroma.chunk_count(),
        "embedding_model": h.chroma.embedding_model or emb_name,
        "chunker": h.chroma.chunker or chunker_key,
        "indexer": mode,
        "ready": h.bm25.ready if bm25_only else h.chroma.ready,
    }


def list_indices() -> dict[str, Any]:
    root = store_root()
    rows = list_indices_detailed(root)
    for r in rows:
        r["indexer"] = read_indexer_mode(str(r["index_id"])) or "chroma"
        if r["indexer"] == "bm25":
            r["chunks"] = make_sparse_store(
                str(r["index_id"]), backend=settings.sparse_backend, store_root=root
            ).chunk_count()
    return {
        "indices": [r["index_id"] for r in rows],
        "files": rows,
        "index_store_dir": str(root / "chroma"),
    }


def delete_index(app: FastAPI, index_id: str) -> None:
    validate_index_id(index_id)
    h = app.state.indices.pop(index_id, None)
    if h is None:
        h = index_handles(app, index_id, cache=False)
    h.chroma.delete_index()
    h.bm25.delete_index()


def remove_source(app: FastAPI, index_id: str, source: str) -> None:
    h = index_handles(app, index_id)
    h.chroma.remove_source(source)
    h.bm25.remove_source(source)
