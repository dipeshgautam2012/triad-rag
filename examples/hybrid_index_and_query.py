"""Hybrid ingest + search: chroma + BM25, shared node store (no HTTP API).

Mirrors orchestration hybrid flow:
  chunk -> chroma.add_chunks + bm25.add_chunks -> RRF search

Run from repo root::

    python examples/hybrid_index_and_query.py
    python examples/hybrid_index_and_query.py --chunker hierarchical
    python examples/hybrid_index_and_query.py --query "keyword search"
"""

import argparse
import sys
from pathlib import Path

RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.chunkers.chunker_factory import make_chunker
from app.chunkers.hierarchical_chunker import HierarchicalChunker
from app.embedders.embedder_factory import make_embedder
from app.hybrid import combine_hybrid_results, format_retrieved, node_from_retrieved
from app.indexers.indexer_factory import make_bm25_indexer, make_chroma_indexer
from app.stores.store_factory import make_node_store, make_sparse_store, make_vector_store

EXAMPLE_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = EXAMPLE_DIR / "sample.txt"
STORE_ROOT = EXAMPLE_DIR / "data" / "index_store"
INDEX_ID = "hybrid"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HYBRID_CANDIDATE_MULTIPLIER = 3


def _chunk_file(path: Path, chunker_name: str):
    embedder = make_embedder(EMBEDDING_MODEL, backend="huggingface")
    chunker = make_chunker(
        chunker_name,
        chunk_size=120,
        chunk_overlap=20,
        embed_model=embedder.embedding_model if chunker_name == "semantic" else None,
        hierarchical_parent_multiplier=3,
        sentence_window_size=3,
    )
    result = chunker.chunk_file(path)
    print(f"chunker={result.chunker!r} chunks={len(result.chunks)}")
    for i, c in enumerate(result.chunks):
        preview = c.text.replace("\n", " ")[:70]
        print(f"  [{i}] {c.chunk_id}: {preview!r}")
    lookup_nodes = (
        chunker.hierarchy_nodes if isinstance(chunker, HierarchicalChunker) else None
    )
    return result, lookup_nodes


def _run_hybrid(result, lookup_nodes, query: str, top_k: int, chunker_name: str, expand: bool) -> None:
    root = STORE_ROOT

    # Same wiring as orchestration.index_handles — one node store, two indexers
    node = make_node_store(INDEX_ID, backend="json", store_root=root)
    embedding_store = make_vector_store(INDEX_ID, backend="chroma", store_root=root)
    keyword_store = make_sparse_store(INDEX_ID, backend="json_bm25", store_root=root)

    embedding_store.delete_store()
    embedding_store.create_collection(embedding_model=EMBEDDING_MODEL, chunker=chunker_name)
    node.delete_store()

    chroma = make_chroma_indexer(
        INDEX_ID,
        embedding_store=embedding_store,
        lookup_store=node,
        embedder=make_embedder(EMBEDDING_MODEL, backend="huggingface"),
    )
    bm25 = make_bm25_indexer(
        INDEX_ID,
        keyword_store=keyword_store,
        context_store=node,
    )
    bm25.delete_index()

    # Same order as orchestration.ingest_file for hybrid (bm25 skips context write)
    n_chroma = chroma.add_chunks(result, lookup_nodes=lookup_nodes)
    n_bm25 = bm25.add_chunks(result, write_context=False)
    print(f"\nindexed chroma={n_chroma} bm25={n_bm25}")
    print(f"  chroma  -> {root / 'chroma'}")
    print(f"  bm25    -> {root / 'sparse' / INDEX_ID}")
    print(f"  nodes   -> {root / 'node_store' / f'{INDEX_ID}.json'}")

    retrieve_k = max(top_k, top_k * HYBRID_CANDIDATE_MULTIPLIER)
    vector_raw = chroma.search(query, retrieve_k, expand=expand)
    sparse_raw = bm25.search(query, retrieve_k, expand=expand)

    print(f"\nquery: {query!r}")
    print("\nchroma hits:")
    for rank, hit in enumerate(vector_raw, start=1):
        print(f"  {rank}. score={hit['score']:.4f}  {hit['chunk_id']!r}  {hit['text'][:60]!r}")

    print("\nbm25 hits:")
    for rank, hit in enumerate(sparse_raw, start=1):
        print(f"  {rank}. score={hit['score']:.4f}  {hit['chunk_id']!r}  {hit['text'][:60]!r}")

    fused = combine_hybrid_results(
        [node_from_retrieved(x) for x in vector_raw],
        sparse_raw,
        limit=retrieve_k,
    )
    print("\nhybrid (RRF):")
    for rank, hit in enumerate(fused[:top_k], start=1):
        row = format_retrieved(hit)
        print(
            f"  {rank}. score={row['score']:.4f}  {row['chunk_id']!r}  {row['text'][:60]!r}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid chunk, index, and search sample.txt")
    p.add_argument(
        "--chunker",
        choices=("simple", "hierarchical", "sentence_window"),
        default="simple",
    )
    p.add_argument("--file", type=Path, default=SAMPLE_FILE)
    p.add_argument("--query", default="How does similarity search work?")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--expand", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if not args.file.is_file():
        sys.exit(f"file not found: {args.file}")

    print(f"file: {args.file}")
    result, lookup_nodes = _chunk_file(args.file, args.chunker)
    _run_hybrid(result, lookup_nodes, args.query, args.top_k, args.chunker, args.expand)


if __name__ == "__main__":
    main()
