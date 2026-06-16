"""Retrieval HTTP API — ingest files, search chunks, manage indices."""

import sys
from pathlib import Path

# python app/main.py from retrieval/ needs the service root on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.chunkers.chunker_factory import CHUNKERS, make_chunker
from app.config import settings
from app.hybrid.combine import combine_hybrid_results
from app.hybrid.merge_hits import format_retrieved, node_from_retrieved, sparse_hit_from_retrieved
from app.indexers.base_indexer import IndexerDeps, validate_index_id
from app.indexers.bm25_indexer import Bm25Indexer
from app.indexers.chroma_indexer import ChromaIndexer
from app.indexers.indexer_factory import make_bm25_indexer, make_chroma_indexer
from app.rerankers.base_reranker import BaseReranker
from app.ingest.filename_sanitizer import sanitize_corpus_filename
from app.ingest.upload import save_upload
from app.stores.base_vector_store import IndexSnapshotError
from app.stores.chroma_vector_store import (
    list_indices_detailed,
    write_index_description,
)
from app.embedders.embedder_factory import make_embedder
from app.rerankers.reranker_factory import make_reranker
from app.stores.store_factory import make_node_store, make_sparse_store, make_vector_store


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    index_id: str = Field(..., min_length=1, description="Logical retrieval index.")
    rerank: bool | None = None


class RetrievedChunk(BaseModel):
    """One passage returned by POST /retrieve.

    score is similarity from vector search (lower distance → higher score).
    metadata is set at ingest: source (filename), file_type, PDF page; hierarchical
    indexes add chunk_role and parent_id. Extra keys are allowed without an API change.
    """

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Ingest-time fields: source, file_type, page, chunk_role, parent_id, …",
    )


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    index_id: str
    candidate_count: int = 0
    candidates: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Pre-rerank candidate pool (debug/assurance); empty when no pool was built.",
    )


class IngestResponse(BaseModel):
    index_id: str
    saved_as: str
    chunks_indexed: int
    embedding_model: str | None = None
    chunker: str | None = None
    indexer: str | None = None
    ready: bool


class IndexDescription(BaseModel):
    description: str = Field(default="", max_length=500)


@dataclass
class IndexHandles:
    chroma: ChromaIndexer
    bm25: Bm25Indexer


def _indexer_deps() -> IndexerDeps:
    return IndexerDeps(
        corpus_dir=Path(settings.corpus_dir).resolve(),
        index_store_dir=Path(settings.index_store_dir).resolve(),
        default_embedding_model=settings.default_embedding_model,
        default_chunker_name=settings.chunker_name,
        rerank_enabled=settings.rerank_enabled,
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        hierarchical_expand_parent=settings.hierarchical_expand_parent,
        hybrid_enabled=settings.hybrid_enabled,
        hybrid_candidate_multiplier=settings.hybrid_candidate_multiplier,
    )


def _index_handles(app: FastAPI, index_id: str, *, cache: bool = True) -> IndexHandles:
    """Chroma + BM25 for one index_id. Reuse app.state.indices when cache=True."""
    if cache:
        cached = app.state.indices.get(index_id)
        if cached is not None:
            return cached
    deps = app.state.indexer_deps
    store_root = deps.index_store_dir
    chroma = make_chroma_indexer(
        index_id,
        deps,
        vector_store=make_vector_store(
            index_id,
            backend=settings.vector_backend,
            store_root=store_root,
        ),
        node_store=make_node_store(
            index_id,
            backend=settings.node_store_backend,
            store_root=store_root,
        ),
    )
    bm25 = make_bm25_indexer(
        index_id,
        deps,
        sparse_store=make_sparse_store(
            index_id,
            backend=settings.sparse_backend,
            store_root=store_root,
        ),
    )
    handles = IndexHandles(chroma=chroma, bm25=bm25)
    if cache:
        app.state.indices[index_id] = handles
    return handles


def _delete_index_storage(handles: IndexHandles) -> None:
    handles.chroma.delete_index()
    handles.bm25.delete_index()


INDEXER_CHOICES = ("vector", "bm25", "hybrid")


def _available_indexers() -> list[str]:
    """Indexers buildable with the configured backends (bm25/hybrid need a sparse backend)."""
    if settings.sparse_backend.strip().lower() == "none":
        return ["vector"]
    return list(INDEXER_CHOICES)


def _index_vector_store(index_id: str, deps: IndexerDeps):
    return make_vector_store(
        index_id,
        backend=settings.vector_backend,
        store_root=deps.index_store_dir,
    )


def _read_indexer_choice(index_id: str, deps: IndexerDeps) -> str | None:
    """The indexer recorded on the collection at ingest, or None if absent."""
    col = _index_vector_store(index_id, deps).try_get_collection()
    if col is None:
        return None
    val = str((col.metadata or {}).get("indexer", "")).strip().lower()
    return val or None


def _write_indexer_choice(index_id: str, deps: IndexerDeps, indexer: str) -> None:
    store = _index_vector_store(index_id, deps)
    col = store.get_collection()
    md = dict(col.metadata or {})
    if md.get("indexer") != indexer:
        md["indexer"] = indexer
        store.modify_metadata(md)


def _search(
    handles: IndexHandles,
    deps: IndexerDeps,
    reranker: BaseReranker,
    query: str,
    top_k: int,
    *,
    rerank: bool | None,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Search with the index's recorded indexer (mode): vector, bm25, or hybrid.

    Returns (final chunks, candidate pool). The pool is the pre-rerank (or
    pre-truncation fused) set, empty when no intermediate pool exists.
    """
    chroma = handles.chroma
    bm25 = handles.bm25
    resolved = mode.strip().lower()
    k = max(1, int(top_k))
    use_rerank = deps.rerank_enabled if rerank is None else rerank
    q = query.strip()

    if resolved == "bm25":
        if not bm25.ready:
            return [], []
        retrieve_k = max(k, k * deps.rerank_candidate_multiplier) if use_rerank else k
        raw = bm25.search(q, retrieve_k)
        if not use_rerank:
            return raw[:k], []
        hits = [node_from_retrieved(h) for h in raw]
        hits = reranker.rerank(hits, q, top_n=k)
        return [format_retrieved(h) for h in hits], raw

    if not chroma.ready:
        return [], []

    if resolved == "hybrid" and bm25.active:
        retrieve_k = k * deps.hybrid_candidate_multiplier
        if use_rerank:
            retrieve_k = max(retrieve_k, k * deps.rerank_candidate_multiplier)
        vector_hits = [node_from_retrieved(h) for h in chroma.search(q, retrieve_k, rerank=False)]
        sparse_hits = [sparse_hit_from_retrieved(h) for h in bm25.search(q, retrieve_k)]
        hits = combine_hybrid_results(vector_hits, sparse_hits, limit=retrieve_k)
        pool = [format_retrieved(h) for h in hits]
        if use_rerank:
            hits = reranker.rerank(hits, q, top_n=k)
            return [format_retrieved(h) for h in hits], pool
        return pool[:k], pool

    if use_rerank:
        retrieve_k = max(k, k * deps.rerank_candidate_multiplier)
        raw = chroma.search(q, retrieve_k, rerank=False)
        hits = [node_from_retrieved(h) for h in raw]
        hits = reranker.rerank(hits, q, top_n=k)
        return [format_retrieved(h) for h in hits], raw
    return chroma.search(q, k, rerank=False), []


def _validate_index_id(index_id: str) -> None:
    try:
        validate_index_id(index_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _preload_embedding_models() -> None:
    """Load every configured embedding model at startup; fail fast if any cannot load."""
    failures: list[str] = []
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
        raise ValueError(
            "available_embedding_models contains models that could not be loaded: "
            + "; ".join(failures)
        )


def _bind_embedder(chroma: ChromaIndexer) -> None:
    if chroma.embedding_model:
        chroma.bind_embedder(
            make_embedder(chroma.embedding_model, backend=settings.embedder_backend)
        )


def _ensure_loaded(handles: IndexHandles) -> None:
    """Load embedding_model, chunker, and description from disk; bind embedder."""
    if handles.chroma.ready or handles.bm25.ready:
        return
    try:
        handles.chroma.load()
        handles.bm25.load()
        _bind_embedder(handles.chroma)
    except IndexSnapshotError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    unknown_chunkers = {c.strip().lower() for c in settings.available_chunkers} - CHUNKERS
    if unknown_chunkers:
        raise ValueError(
            f"available_chunkers has unimplemented entries: {sorted(unknown_chunkers)}; "
            f"implemented: {sorted(CHUNKERS)}"
        )
    _preload_embedding_models()
    Path(settings.corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.index_store_dir).mkdir(parents=True, exist_ok=True)
    app.state.indices = {}
    app.state.indexer_deps = _indexer_deps()
    app.state.reranker = make_reranker(
        settings.rerank_model, backend=settings.reranker_backend
    )
    handles = _index_handles(app, "default")
    try:
        handles.chroma.load()
        handles.bm25.load()
        _bind_embedder(handles.chroma)
    except IndexSnapshotError as e:
        logging.error("Default index load failed: %s", e)
        raise
    yield
    app.state.indices.clear()


app = FastAPI(title="Triad RAG Retrieval", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "retrieval", "status": "ok"}


@app.get("/indices")
def list_indices(request: Request) -> dict[str, Any]:
    """List saved indices and per-index metadata (embedding model, chunker, chunk count)."""
    deps = request.app.state.indexer_deps
    rows = list_indices_detailed(deps.index_store_dir)
    for r in rows:
        r["indexer"] = _read_indexer_choice(str(r["index_id"]), deps) or "vector"
        if r["indexer"] == "bm25":
            # Vector count is 0 for a bm25-only index; report the sparse count.
            r["chunks"] = make_sparse_store(
                str(r["index_id"]),
                backend=settings.sparse_backend,
                store_root=deps.index_store_dir,
            ).chunk_count()
    store = (Path(settings.index_store_dir).resolve() / "chroma").resolve()
    return {
        "indices": [r["index_id"] for r in rows],
        "files": rows,
        "index_store_dir": str(store),
    }


@app.get("/ingest/options")
def ingest_options() -> dict[str, Any]:
    """Choices for a new ingest (indexer, embedding model, chunker) plus fixed backends."""
    return {
        "indexers": _available_indexers(),
        "default_indexer": "vector",
        "embedding_models": settings.available_embedding_models,
        "default_embedding_model": settings.default_embedding_model,
        "chunkers": settings.available_chunkers,
        "default_chunker": settings.chunker_name,
        "embedder_backend": settings.embedder_backend,
        "sparse_backend": settings.sparse_backend,
    }


@app.post("/indices/{index_id}/description")
def set_index_description(
    request: Request, index_id: str, body: IndexDescription
) -> dict[str, Any]:
    """Save index description without re-embedding."""
    _validate_index_id(index_id)
    try:
        md = write_index_description(
            index_id, body.description, request.app.state.indexer_deps.index_store_dir
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"no saved index for index_id={index_id!r}"
        ) from None
    handles = request.app.state.indices.get(index_id)
    if handles is not None:
        handles.chroma.index_metadata = dict(md)
    return {"index_id": index_id, "index_metadata": md}


@app.delete("/indices/{index_id}")
def delete_index(request: Request, index_id: str) -> dict[str, Any]:
    """Evict from memory and remove the saved index."""
    _validate_index_id(index_id)
    handles = request.app.state.indices.pop(index_id, None)
    if handles is not None:
        _delete_index_storage(handles)
    else:
        _delete_index_storage(_index_handles(request.app, index_id, cache=False))
    return {"index_id": index_id, "deleted": True}


@app.get("/indices/{index_id}/files")
def list_corpus_files(request: Request, index_id: str) -> dict[str, Any]:
    """List ``.txt`` / ``.pdf`` files in this index's corpus folder."""
    _validate_index_id(index_id)
    handles = _index_handles(request.app, index_id)
    return {"index_id": index_id, "files": handles.chroma.list_corpus_files()}


@app.delete("/indices/{index_id}/corpus")
def clear_corpus(request: Request, index_id: str) -> dict[str, Any]:
    """Delete all corpus files for this index (not the Chroma index)."""
    _validate_index_id(index_id)
    handles = _index_handles(request.app, index_id)
    deleted: list[str] = []
    for name in handles.chroma.list_corpus_files():
        handles.bm25.delete_by_source(name)
        deleted.append(handles.chroma.delete_corpus_file(name))
    return {"index_id": index_id, "deleted": deleted}


@app.delete("/indices/{index_id}/files/{filename}")
def delete_corpus_file(request: Request, index_id: str, filename: str) -> dict[str, Any]:
    """Delete one corpus file and remove its chunks from the index."""
    _validate_index_id(index_id)
    handles = _index_handles(request.app, index_id)
    try:
        name = sanitize_corpus_filename(filename)
        handles.bm25.delete_by_source(Path(name).name)
        deleted = handles.chroma.delete_corpus_file(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"no corpus file {filename!r} for index_id={index_id!r}"
        ) from None
    return {"index_id": index_id, "deleted": deleted}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: Request, body: RetrieveRequest) -> RetrieveResponse:
    _validate_index_id(body.index_id)
    handles = _index_handles(request.app, body.index_id)
    _ensure_loaded(handles)
    mode = _read_indexer_choice(body.index_id, request.app.state.indexer_deps) or "vector"
    chunks_raw, candidates_raw = _search(
        handles,
        request.app.state.indexer_deps,
        request.app.state.reranker,
        body.query,
        body.top_k,
        rerank=body.rerank,
        mode=mode,
    )
    chunks = [RetrievedChunk.model_validate(c) for c in chunks_raw]
    candidates = [RetrievedChunk.model_validate(c) for c in candidates_raw]
    return RetrieveResponse(
        query=body.query,
        chunks=chunks,
        index_id=body.index_id,
        candidate_count=len(candidates),
        candidates=candidates,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    index_id: str = Form(...),
    index_description: str | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    chunker_name: str | None = Form(default=None),
    indexer: str | None = Form(default=None),
) -> IngestResponse:
    _validate_index_id(index_id)
    deps = request.app.state.indexer_deps

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

    available_indexers = _available_indexers()
    requested_indexer = (indexer or "").strip().lower()
    if requested_indexer and requested_indexer not in available_indexers:
        raise HTTPException(
            status_code=400,
            detail=f"indexer must be one of {available_indexers}",
        )
    recorded_indexer = _read_indexer_choice(index_id, deps)
    if recorded_indexer and requested_indexer and requested_indexer != recorded_indexer:
        raise HTTPException(
            status_code=409,
            detail=(
                f"index {index_id!r} was created with indexer {recorded_indexer!r}; "
                f"cannot ingest with {requested_indexer!r}"
            ),
        )
    resolved_indexer = requested_indexer or recorded_indexer or "vector"
    if resolved_indexer not in available_indexers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"index {index_id!r} uses indexer {resolved_indexer!r}, "
                f"but the server sparse_backend is 'none'"
            ),
        )

    data = await file.read()
    handles = _index_handles(request.app, index_id)
    corpus_path = handles.chroma.corpus_dir()
    try:
        saved = save_upload(
            data,
            file.filename,
            corpus_path,
            max_bytes=settings.max_upload_bytes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    resolved_embedding_model = (embedding_model or "").strip() or settings.default_embedding_model
    resolved_chunker = (chunker_name or "").strip() or settings.chunker_name
    ingest_embedder = make_embedder(
        resolved_embedding_model,
        backend=settings.embedder_backend,
    )
    chunker = make_chunker(
        resolved_chunker,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embed_model=ingest_embedder.embedding_model if resolved_chunker == "semantic" else None,
        hierarchical_parent_multiplier=settings.hierarchical_parent_multiplier,
        sentence_window_size=settings.sentence_window_size,
        semantic_breakpoint_percentile=settings.semantic_breakpoint_percentile,
        semantic_buffer_size=settings.semantic_buffer_size,
    )
    chunks = chunker.chunk_file(saved)
    try:
        if resolved_indexer in ("vector", "hybrid"):
            handles.chroma.add_chunks(
                chunks,
                source=saved.name,
                embedder=ingest_embedder,
                description=index_description,
                chunker_name=chunker_name,
            )
        else:
            # bm25-only: no vector chunks, but the collection still anchors the
            # index in /indices and carries its metadata (chunker, indexer, …).
            store = _index_vector_store(index_id, deps)
            if store.try_get_collection() is None:
                store.create_collection(
                    embedding_model=resolved_embedding_model,
                    chunker=resolved_chunker,
                    description=(index_description or "").strip()[:500] or None,
                )
            elif chunker_name and resolved_chunker != store.resolve_chunker():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"index {index_id!r} uses chunker {store.resolve_chunker()!r}; "
                        f"cannot ingest with {resolved_chunker!r}"
                    ),
                )
        if resolved_indexer in ("bm25", "hybrid"):
            handles.bm25.add_chunks(
                chunks,
                source=saved.name,
                embedder=ingest_embedder,
                description=index_description,
                chunker_name=chunker_name,
            )
    except IndexSnapshotError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    _write_indexer_choice(index_id, deps, resolved_indexer)

    bm25_only = resolved_indexer == "bm25"
    return IngestResponse(
        index_id=index_id,
        saved_as=saved.name,
        chunks_indexed=handles.bm25.chunk_count() if bm25_only else handles.chroma.chunk_count(),
        embedding_model=handles.chroma.embedding_model or resolved_embedding_model,
        chunker=handles.chroma.chunker or resolved_chunker,
        indexer=resolved_indexer,
        ready=handles.bm25.ready if bm25_only else handles.chroma.ready,
    )


def _parse_run_args(default_host: str, argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the retrieval ASGI app (dev server).")
    p.add_argument(
        "--host",
        default=default_host,
        help=f"bind host (default: {default_host})",
    )
    p.add_argument(
        "--no-reload",
        action="store_true",
        help="disable autoreload (default: reload on)",
    )
    return p.parse_args(argv)


def run(host: str, port: int, *, reload: bool = True) -> None:
    """Start the dev server (same ASGI app as ``uvicorn app.main:app``)."""
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


def main(argv: list[str] | None = None) -> None:
    bind_host = "127.0.0.1"
    port = 8101
    args = _parse_run_args(bind_host, argv)
    run(args.host, port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
