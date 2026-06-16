"""Vector index (Chroma) — embed, store, and similarity search."""

import logging
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaVectorStore

from app.chunkers.base_chunker import ChunkSet
from app.embedders.base_embedder import BaseEmbedder
from app.indexers.base_indexer import BaseIndexer, IndexerDeps, validate_index_id
from app.rerankers.base_reranker import BaseReranker
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_vector_store import IndexSnapshotError, BaseVectorStore

logger = logging.getLogger(__name__)


def _format_search_result(hit: NodeWithScore) -> dict[str, Any]:
    node = hit.node
    return {
        "chunk_id": node.node_id,
        "text": node.get_content(),
        "score": float(hit.score or 0.0),
        "metadata": dict(node.metadata or {}),
    }


def _storage_context(
    vector_store: BaseVectorStore,
    node_store: BaseNodeStore,
) -> StorageContext:
    col = vector_store.get_collection()
    vs = LlamaChromaVectorStore(chroma_collection=col)
    if node_store.exists():
        return StorageContext.from_defaults(
            vector_store=vs, docstore=node_store.as_llama_docstore()
        )
    return StorageContext.from_defaults(vector_store=vs)


def _vector_index(
    vector_store: BaseVectorStore,
    node_store: BaseNodeStore,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    col = vector_store.get_collection()
    vs = LlamaChromaVectorStore(chroma_collection=col)
    ctx = _storage_context(vector_store, node_store)
    return VectorStoreIndex.from_vector_store(
        vs,
        storage_context=ctx,
        embed_model=embed_model,
    )


class ChromaIndexer(BaseIndexer):
    """One named index — vector ingest, search, and corpus lifecycle."""

    def __init__(
        self,
        index_id: str,
        deps: IndexerDeps,
        *,
        vector_store: BaseVectorStore,
        node_store: BaseNodeStore,
    ) -> None:
        validate_index_id(index_id)
        self.index_id = index_id
        self._deps = deps
        self._vector_store = vector_store
        self._node_store = node_store
        self._embedder: BaseEmbedder | None = None
        self._reranker: BaseReranker | None = None
        self._loaded = False
        self.index_metadata: dict[str, Any] = {}
        self.embedding_model: str | None = None
        self.chunker: str | None = None

    def bind_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def bind_reranker(self, reranker: BaseReranker) -> None:
        self._reranker = reranker

    def _require_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            raise RuntimeError("embedder not bound; call bind_embedder() from main first")
        return self._embedder

    def corpus_dir(self) -> Path:
        root = self._deps.corpus_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if self.index_id == "default":
            return root
        sub = (root / self.index_id).resolve()
        try:
            sub.relative_to(root)
        except ValueError as e:
            raise ValueError("invalid index_id path") from e
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def delete_index(self) -> None:
        self._vector_store.delete_store()
        self._node_store.delete_store()
        self.index_metadata = {}
        self.embedding_model = None
        self.chunker = None
        self._embedder = None
        self._loaded = False

    def list_corpus_files(self) -> list[str]:
        corpus = self.corpus_dir()
        names = {p.name for p in [*corpus.glob("*.txt"), *corpus.glob("*.pdf")]}
        return sorted(names)

    def delete_corpus_file(self, filename: str) -> str:
        name = Path(filename).name
        corpus = self.corpus_dir()
        dest = (corpus / name).resolve()
        try:
            dest.relative_to(corpus.resolve())
        except ValueError as e:
            raise ValueError("Invalid path") from e
        if not dest.is_file():
            raise FileNotFoundError(name)

        if self._vector_store.exists():
            self._vector_store.delete_by_source(name)
            self._node_store.delete_by_source(name)

        dest.unlink()
        if not self.list_corpus_files() and self.index_id != "default":
            try:
                corpus.rmdir()
            except OSError:
                pass
        logger.info("Deleted corpus file %s for index %s", name, self.index_id)
        return name

    def clear_corpus(self) -> list[str]:
        removed: list[str] = []
        for name in self.list_corpus_files():
            self.delete_corpus_file(name)
            removed.append(name)
        return removed

    def load(self) -> None:
        if self._vector_store.chunk_count() == 0:
            return
        model = self._vector_store.resolve_embedding_model()
        self.embedding_model = model
        self.chunker = self._vector_store.resolve_chunker()
        desc = self._vector_store.read_description()
        self.index_metadata = {"description": desc} if desc else {}
        self._loaded = True

    def add_chunks(
        self,
        chunks: ChunkSet,
        *,
        source: str,
        embedder: BaseEmbedder,
        description: str | None = None,
        chunker_name: str | None = None,
    ) -> int:
        if not chunks.embed_chunks:
            return 0

        vector_store = self._vector_store
        col = vector_store.try_get_collection()
        cleaned_desc = (description or "").strip()[:500]
        use_node_store = chunks.all_chunks is not None
        requested_chunker = (chunker_name or "").strip()
        self._embedder = embedder

        if col is None:
            model = embedder.model_name
            chunker = requested_chunker or self._deps.default_chunker_name
            vector_store.create_collection(
                embedding_model=model,
                chunker=chunker,
                description=cleaned_desc or None,
            )
        else:
            model = vector_store.resolve_embedding_model()
            chunker = vector_store.resolve_chunker()
            if embedder.model_name != model:
                raise IndexSnapshotError(
                    f"index {self.index_id!r} uses embedding_model {model!r}; "
                    f"cannot ingest with {embedder.model_name!r}"
                )
            if requested_chunker and requested_chunker != chunker:
                raise IndexSnapshotError(
                    f"index {self.index_id!r} uses chunker {chunker!r}; "
                    f"cannot ingest with {requested_chunker!r}"
                )
            vector_store.delete_by_source(source)
            if use_node_store:
                self._node_store.delete_by_source(source)

            new_md = dict(col.metadata or {})
            new_md["embedding_model"] = model
            new_md["chunker"] = chunker
            if description is not None:
                new_md.pop("description", None)
                if cleaned_desc:
                    new_md["description"] = cleaned_desc
            vector_store.modify_metadata(new_md)

        if col is None or description is not None:
            self.index_metadata = {"description": cleaned_desc} if cleaned_desc else {}

        if use_node_store:
            self._node_store.add_nodes(chunks.all_chunks)

        index = _vector_index(vector_store, self._node_store, embedder.embedding_model)
        index.insert_nodes(chunks.embed_chunks)
        self.embedding_model = model
        self.chunker = chunker
        self._loaded = True
        return len(chunks.embed_chunks)

    @property
    def ready(self) -> bool:
        if not self._loaded:
            return False
        return self._vector_store.chunk_count() > 0

    def chunk_count(self) -> int:
        return self._vector_store.chunk_count()

    def search(self, query: str, top_k: int, *, rerank: bool | None = None) -> list[dict[str, Any]]:
        """Vector search, then optional rerank. top_k is final count."""
        if not self._loaded:
            return []
        if self._vector_store.chunk_count() == 0:
            return []

        embedder = self._require_embedder()
        index = _vector_index(self._vector_store, self._node_store, embedder.embedding_model)
        k = max(1, int(top_k))
        use_rerank = self._deps.rerank_enabled if rerank is None else rerank
        retrieve_k = k
        if use_rerank:
            retrieve_k = max(retrieve_k, k * self._deps.rerank_candidate_multiplier)
        vector_retriever = VectorIndexRetriever(index, similarity_top_k=retrieve_k)
        if self._deps.hierarchical_expand_parent and self._node_store.exists():
            ctx = _storage_context(self._vector_store, self._node_store)
            retriever = AutoMergingRetriever(
                vector_retriever,
                ctx,
                simple_ratio_thresh=0.4,
                verbose=False,
            )
        else:
            retriever = vector_retriever

        q = query.strip()
        hits = retriever.retrieve(q)
        if use_rerank:
            if self._reranker is None:
                raise RuntimeError("reranker not bound; call bind_reranker() from main first")
            hits = self._reranker.rerank(hits, q, top_n=k)
        return [_format_search_result(h) for h in hits]
