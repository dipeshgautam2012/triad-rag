"""Minimal end-to-end: chunk -> index -> search (no HTTP API).

Run from repo root::

    python examples/index_and_query.py
    python examples/index_and_query.py --indexer bm25
    python examples/index_and_query.py --query "keyword search"
"""

import argparse
import sys
from pathlib import Path

RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.chunkers.simple_chunker import SimpleChunker
from app.embedders.embedder_factory import make_embedder
from app.indexers.indexer_factory import make_bm25_indexer, make_chroma_indexer
from app.stores.store_factory import make_node_store, make_sparse_store, make_vector_store

EXAMPLE_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = EXAMPLE_DIR / "sample.txt"
STORE_ROOT = EXAMPLE_DIR / "data" / "index_store"
INDEX_ID = "example"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _chunk_file(path: Path):
    chunker = SimpleChunker(chunk_size=120, chunk_overlap=20)
    result = chunker.chunk_file(path)
    print(f"chunker={result.chunker!r} chunks={len(result.chunks)}")
    for i, c in enumerate(result.chunks):
        preview = c.text.replace("\n", " ")[:70]
        print(f"  [{i}] {c.chunk_id}: {preview!r}")
    return result


def _run_chroma(result, query: str, top_k: int) -> None:
    root = STORE_ROOT

    # 1. Stores — where vectors and lookup nodes live on disk
    embedding_store = make_vector_store(INDEX_ID, backend="chroma", store_root=root)
    lookup_store = make_node_store(INDEX_ID, backend="json", store_root=root)

    # 2. Empty Chroma collection (indexer does not create this — same as main/orchestration)
    embedding_store.delete_store()
    embedding_store.create_collection(embedding_model=EMBEDDING_MODEL, chunker="simple")
    lookup_store.delete_store()

    # 3. Indexer — save chunks + search, using the stores above
    indexer = make_chroma_indexer(
        INDEX_ID,
        embedding_store=embedding_store,
        lookup_store=lookup_store,
        embedder=make_embedder(EMBEDDING_MODEL, backend="huggingface"),
    )

    n = indexer.add_chunks(result)
    print(f"\nindexed {n} chunks -> {root / 'chroma'}")

    hits = indexer.search(query, top_k)
    print(f"\nquery: {query!r}")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. score={hit['score']:.4f}  {hit['text'][:80]!r}")


def _run_bm25(result, query: str, top_k: int) -> None:
    root = STORE_ROOT
    keyword_store = make_sparse_store(INDEX_ID, backend="json_bm25", store_root=root)
    context_store = make_node_store(INDEX_ID, backend="json", store_root=root)

    indexer = make_bm25_indexer(
        INDEX_ID,
        keyword_store=keyword_store,
        context_store=context_store,
    )
    indexer.delete_index()

    n = indexer.add_chunks(result)
    print(f"\nindexed {n} chunks -> {root / 'sparse' / INDEX_ID}")

    hits = indexer.search(query, top_k)
    print(f"\nquery: {query!r}")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. score={hit['score']:.4f}  {hit['text'][:80]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Chunk, index, and search sample.txt")
    p.add_argument("--indexer", choices=("chroma", "bm25"), default="chroma")
    p.add_argument("--file", type=Path, default=SAMPLE_FILE)
    p.add_argument("--query", default="How does similarity search work?")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    if not args.file.is_file():
        sys.exit(f"file not found: {args.file}")

    print(f"file: {args.file}")
    result = _chunk_file(args.file)

    if args.indexer == "chroma":
        _run_chroma(result, args.query, args.top_k)
    else:
        _run_bm25(result, args.query, args.top_k)


if __name__ == "__main__":
    main()
