"""Build indexers from injected store instances. Called from main only."""

from app.embedders.base_embedder import BaseEmbedder
from app.indexers.bm25_indexer import Bm25Indexer
from app.indexers.chroma_indexer import ChromaIndexer
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_sparse_store import BaseSparseStore
from app.stores.base_vector_store import BaseVectorStore


def make_chroma_indexer(
    index_id: str,
    *,
    embedding_store: BaseVectorStore,
    lookup_store: BaseNodeStore,
    embedder: BaseEmbedder | None = None,
) -> ChromaIndexer:
    indexer = ChromaIndexer(
        index_id,
        embedding_store=embedding_store,
        lookup_store=lookup_store,
    )
    if embedder is not None:
        indexer.bind_embedder(embedder)
    return indexer


def make_bm25_indexer(
    index_id: str,
    *,
    keyword_store: BaseSparseStore,
    context_store: BaseNodeStore,
) -> Bm25Indexer:
    return Bm25Indexer(
        index_id,
        keyword_store=keyword_store,
        context_store=context_store,
    )
