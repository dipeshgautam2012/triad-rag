"""BM25 keyword index — ingest searchable text, search by keywords."""

from typing import Any

import bm25s

from app.chunkers.base_chunker import ChunkingResult
from app.indexers.base_indexer import (
    BaseIndexer,
    context_nodes,
    _expand_hit_context,
    format_search_hit,
    searchable_nodes,
)
from app.stores.base_node_store import BaseNodeStore
from app.stores.base_sparse_store import BaseSparseStore


class Bm25Indexer(BaseIndexer):
    """BM25 ingest and keyword search.

    keyword_store — searchable text.
    context_store — context for lookup at search time; not searched.
    """

    def __init__(
        self,
        index_id: str,
        *,
        keyword_store: BaseSparseStore,
        context_store: BaseNodeStore,
    ) -> None:
        super().__init__(index_id)
        self._keyword_store = keyword_store  # searchable text
        self._context_store = context_store  # context for lookup; not searched

    def remove_source(self, source: str) -> None:
        self._keyword_store.delete_by_source(source)
        self._context_store.delete_by_source(source)

    def delete_index(self) -> None:
        self._keyword_store.delete_store()
        self._context_store.delete_store()
        self._loaded = False

    def load(self) -> None:
        if self._keyword_store.chunk_count() == 0:
            return
        self._loaded = True

    def add_chunks(self, result: ChunkingResult, *, write_context: bool = True) -> int:
        if not result.chunks:
            return 0
        context_for_lookup = context_nodes(result.chunks)
        searchable = searchable_nodes(result.chunks)  # written to keyword_store
        if write_context and context_for_lookup is not None:
            self._context_store.add_nodes(context_for_lookup)
        self._keyword_store.add_chunks(searchable)
        self._loaded = True
        return len(searchable)

    @property
    def ready(self) -> bool:
        if not self._loaded:
            return False
        return self._keyword_store.chunk_count() > 0

    def chunk_count(self) -> int:
        return self._keyword_store.chunk_count()

    def search(self, query: str, top_k: int, *, expand: bool = False) -> list[dict[str, Any]]:
        if not self._loaded:
            return []
        q = query.strip()
        if not q:
            return []
        retriever = self._keyword_store.load_retriever()
        if retriever is None:
            return []
        records = self._keyword_store.load_records()
        if not records:
            return []
        k = max(1, int(top_k))
        query_tokens = bm25s.tokenize(q)
        doc_indices, scores = retriever.retrieve(query_tokens, k=min(k, len(records)))
        indices = doc_indices[0]
        row_scores = scores[0]
        out: list[dict[str, Any]] = []
        for idx, score in zip(indices, row_scores, strict=False):
            i = int(idx)
            if i < 0 or i >= len(records):
                continue
            rec = records[i]
            md = dict(rec.get("metadata") or {})
            out.append(
                format_search_hit(
                    chunk_id=str(rec["chunk_id"]),
                    text=str(rec["text"]),
                    score=float(score),
                    metadata=md,
                )
            )
        if expand:
            out = _expand_hit_context(out, self._context_store)
        return out
