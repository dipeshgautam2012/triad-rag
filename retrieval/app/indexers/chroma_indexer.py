"""Vector index (Chroma) — embed searchable text, similarity search."""

from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaVectorStore

from app.chunkers.base_chunker import ChunkingResult
from app.embedders.base_embedder import BaseEmbedder
from app.indexers.base_indexer import (
    BaseIndexer,
    context_nodes,
    _expand_hit_context,
    format_search_hit,
    searchable_nodes,
)
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_vector_store import BaseVectorStore


def _vector_index(
    embedding_store: BaseVectorStore,
    lookup_store: BaseNodeStore,
    embed_model: BaseEmbedding,
) -> tuple[VectorStoreIndex, StorageContext]:
    col = embedding_store.get_collection()
    vs = LlamaChromaVectorStore(chroma_collection=col)
    if lookup_store.exists():
        ctx = StorageContext.from_defaults(
            vector_store=vs,
            docstore=lookup_store.as_llama_docstore(),
        )
    else:
        ctx = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_vector_store(
        vs,
        storage_context=ctx,
        embed_model=embed_model,
    )
    return index, ctx


class ChromaIndexer(BaseIndexer):
    """Vector ingest and similarity search.

    embedding_store — searchable text (embedded).
    lookup_store — extra text for lookup at search time; not embedded.
    """

    def __init__(
        self,
        index_id: str,
        *,
        embedding_store: BaseVectorStore,
        lookup_store: BaseNodeStore,
    ) -> None:
        super().__init__(index_id)
        self._embedding_store = embedding_store
        self._lookup_store = lookup_store
        self._embedder: BaseEmbedder | None = None
        self.index_metadata: dict[str, Any] = {}
        self.embedding_model: str | None = None
        self.chunker: str | None = None

    def bind_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def _require_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            raise RuntimeError("embedder not bound; pass embedder to make_chroma_indexer or bind_embedder")
        return self._embedder

    def remove_source(self, source: str) -> None:
        self._embedding_store.delete_by_source(source)
        self._lookup_store.delete_by_source(source)

    def delete_index(self) -> None:
        self._embedding_store.delete_store()
        self._lookup_store.delete_store()
        self.index_metadata = {}
        self.embedding_model = None
        self.chunker = None
        self._embedder = None
        self._loaded = False

    def load(self) -> None:
        if self._embedding_store.chunk_count() == 0:
            return
        
        # load index metadata from embedding store
        self.embedding_model = self._embedding_store.resolve_embedding_model()
        self.chunker = self._embedding_store.resolve_chunker()
        desc = self._embedding_store.read_description()
        self.index_metadata = {"description": desc} if desc else {}
        self._loaded = True

    def add_chunks(
        self,
        result: ChunkingResult,
        *,
        lookup_nodes: list[BaseNode] | None = None,
    ) -> int:
        if not result.chunks:
            return 0
        # lookup_nodes are the nodes that are used for lookup at search time; not embedded.
        # window for sentence-window chunks and full hierarchy for hierarchical chunks.
        # no lookup for simple chunkers, markdown and semantic chunkers
        for_lookup = lookup_nodes if lookup_nodes is not None else context_nodes(result.chunks)
        searchable = searchable_nodes(result.chunks)  # written to embedding_store
        if for_lookup is not None:
            # write to lookup store for search time lookup
            self._lookup_store.add_nodes(for_lookup)
        
        
        index, _ = _vector_index(
            self._embedding_store,
            self._lookup_store,
            self._require_embedder().embedding_model,
        )
        # write to embedding store for search time similarity search
        index.insert_nodes(searchable)
        self._loaded = True
        return len(searchable)

    @property
    def ready(self) -> bool:
        if not self._loaded:
            return False
        return self._embedding_store.chunk_count() > 0

    def chunk_count(self) -> int:
        return self._embedding_store.chunk_count()

    def search(
        self,
        query: str,
        top_k: int,
        *,
        expand: bool = False,
    ) -> list[dict[str, Any]]:
        if not self._loaded:
            return []
        q = query.strip()
        if not q or self._embedding_store.chunk_count() == 0:
            return []

        index, ctx = _vector_index(
            self._embedding_store,
            self._lookup_store,
            self._require_embedder().embedding_model,
        )
        k = max(1, int(top_k))
        vector_retriever = VectorIndexRetriever(index, similarity_top_k=k)
        if expand and self._lookup_store.exists():
            retriever: VectorIndexRetriever | AutoMergingRetriever = AutoMergingRetriever(
                vector_retriever,
                ctx,
                simple_ratio_thresh=0.4,
                verbose=False,
            )
        else:
            retriever = vector_retriever

        out: list[dict[str, Any]] = []
        for hit in retriever.retrieve(q):
            md = dict(hit.node.metadata or {})
            out.append(
                format_search_hit(
                    chunk_id=hit.node.node_id,
                    text=hit.node.get_content(),
                    score=float(hit.score or 0.0),
                    metadata=md,
                )
            )
        if expand:
            out = _expand_hit_context(out, self._lookup_store)
        return out
