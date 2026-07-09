"""Pick store implementation from backend name passed by main. Called from main only."""

from pathlib import Path

from app.indexers.base_indexer import validate_index_id
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_sparse_store import BaseSparseStore
from app.stores.base_vector_store import BaseVectorStore
from app.stores.chroma_vector_store import ChromaVectorStore
from app.stores.json_bm25_sparse_store import JsonBm25SparseStore
from app.stores.json_node_store import JsonNodeStore
from app.stores.none_sparse_store import NoneSparseStore
from app.stores.sqlite_bm25_sparse_store import SqliteBm25SparseStore
from app.stores.sqlite_node_store import SqliteNodeStore

VECTOR_BACKENDS = frozenset({"chroma"})
NODE_STORE_BACKENDS = frozenset({"json", "sqlite"})
SPARSE_BACKENDS = frozenset({"none", "json_bm25", "sqlite_bm25"})


def make_vector_store(
    index_id: str,
    *,
    backend: str,
    store_root: Path,
) -> BaseVectorStore:
    validate_index_id(index_id)
    key = backend.strip().lower()
    if key not in VECTOR_BACKENDS:
        raise ValueError(f"unsupported vector backend: {backend!r}; supported: {sorted(VECTOR_BACKENDS)}")
    root = store_root.resolve()
    return ChromaVectorStore(index_id, store_root=root)


def make_node_store(
    index_id: str,
    *,
    backend: str,
    store_root: Path,
) -> BaseNodeStore:
    validate_index_id(index_id)
    key = backend.strip().lower()
    if key not in NODE_STORE_BACKENDS:
        raise ValueError(f"unsupported node_store backend: {backend!r}; supported: {sorted(NODE_STORE_BACKENDS)}")
    root = store_root.resolve()
    if key == "json":
        return JsonNodeStore(index_id, store_root=root)
    return SqliteNodeStore(index_id, store_root=root)


def make_sparse_store(
    index_id: str,
    *,
    backend: str,
    store_root: Path,
) -> BaseSparseStore:
    validate_index_id(index_id)
    key = backend.strip().lower()
    if key not in SPARSE_BACKENDS:
        raise ValueError(f"unsupported sparse backend: {backend!r}; supported: {sorted(SPARSE_BACKENDS)}")
    root = store_root.resolve()
    if key == "none":
        return NoneSparseStore()
    if key == "json_bm25":
        return JsonBm25SparseStore(index_id, store_root=root)
    return SqliteBm25SparseStore(index_id, store_root=root)
