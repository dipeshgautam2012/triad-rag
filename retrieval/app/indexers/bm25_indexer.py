"""BM25 keyword index — store chunks, build sparse index, search."""

import logging
from pathlib import Path
from typing import Any

import bm25s

from app.chunkers.base_chunker import ChunkSet
from app.embedders.base_embedder import BaseEmbedder
from app.indexers.base_indexer import BaseIndexer, IndexerDeps, validate_index_id
from app.rerankers.base_reranker import BaseReranker
from app.stores.base_sparse_store import BaseSparseStore, SparseHit

logger = logging.getLogger(__name__)


def _format_sparse_hit(hit: SparseHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "text": hit.text,
        "score": hit.score,
        "metadata": dict(hit.metadata),
    }


class Bm25Indexer(BaseIndexer):
    """One named index — BM25 ingest, search, and corpus lifecycle."""

    def __init__(
        self,
        index_id: str,
        deps: IndexerDeps,
        *,
        sparse_store: BaseSparseStore,
    ) -> None:
        validate_index_id(index_id)
        self.index_id = index_id
        self._deps = deps
        self._sparse_store = sparse_store
        self._loaded = False
        self.index_metadata: dict[str, Any] = {}
        self.embedding_model: str | None = None
        self.chunker: str | None = None

    def bind_embedder(self, embedder: BaseEmbedder) -> None:
        return None

    def bind_reranker(self, reranker: BaseReranker) -> None:
        return None

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

    @property
    def active(self) -> bool:
        return self._sparse_store.active

    def delete_by_source(self, source: str) -> None:
        self._sparse_store.delete_by_source(source)

    def delete_index(self) -> None:
        self._sparse_store.delete_store()
        self.index_metadata = {}
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

        if self._sparse_store.chunk_count() > 0:
            self._sparse_store.delete_by_source(name)

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
        if self._sparse_store.chunk_count() == 0:
            return
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
        if not self._sparse_store.active:
            return 0
        self.delete_by_source(source)
        self._sparse_store.add_chunks(chunks.embed_chunks)
        self._loaded = True
        return len(chunks.embed_chunks)

    @property
    def ready(self) -> bool:
        if not self._loaded:
            return False
        return self._sparse_store.active and self._sparse_store.chunk_count() > 0

    def chunk_count(self) -> int:
        return self._sparse_store.chunk_count()

    def search(self, query: str, top_k: int, *, rerank: bool | None = None) -> list[dict[str, Any]]:
        if not self._loaded or not self._sparse_store.active:
            return []
        q = query.strip()
        if not q:
            return []
        retriever = self._sparse_store.load_retriever()
        if retriever is None:
            return []
        records = self._sparse_store.load_records()
        if not records:
            return []
        k = max(1, int(top_k))
        query_tokens = bm25s.tokenize(q)
        doc_indices, scores = retriever.retrieve(query_tokens, k=min(k, len(records)))
        indices = doc_indices[0]
        row_scores = scores[0]
        hits: list[SparseHit] = []
        for idx, score in zip(indices, row_scores, strict=False):
            i = int(idx)
            if i < 0 or i >= len(records):
                continue
            rec = records[i]
            hits.append(
                SparseHit(
                    chunk_id=str(rec["chunk_id"]),
                    text=str(rec["text"]),
                    score=float(score),
                    metadata=dict(rec.get("metadata") or {}),
                )
            )
        return [_format_sparse_hit(h) for h in hits]
