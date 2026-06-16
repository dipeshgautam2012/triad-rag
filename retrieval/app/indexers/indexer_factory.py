"""Build indexers from injected store instances. Called from main only."""

from app.indexers.base_indexer import IndexerDeps
from app.indexers.bm25_indexer import Bm25Indexer
from app.indexers.chroma_indexer import ChromaIndexer
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_sparse_store import BaseSparseStore
from app.stores.base_vector_store import BaseVectorStore


def make_chroma_indexer(
    index_id: str,
    deps: IndexerDeps,
    *,
    vector_store: BaseVectorStore,
    node_store: BaseNodeStore,
) -> ChromaIndexer:
    return ChromaIndexer(
        index_id,
        deps,
        vector_store=vector_store,
        node_store=node_store,
    )


def make_bm25_indexer(
    index_id: str,
    deps: IndexerDeps,
    *,
    sparse_store: BaseSparseStore,
) -> Bm25Indexer:
    return Bm25Indexer(
        index_id,
        deps,
        sparse_store=sparse_store,
    )
