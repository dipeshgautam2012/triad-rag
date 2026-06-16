"""Placeholder when keyword search is off (sparse_backend = none)."""

from typing import Any, Sequence

from llama_index.core.schema import BaseNode

from app.stores.base_sparse_store import BaseSparseStore


class NoneSparseStore(BaseSparseStore):
    """No-op when sparse_backend = none — hybrid falls back to vector search only."""

    @property
    def active(self) -> bool:
        return False

    def delete_store(self) -> None:
        return None

    def delete_by_source(self, source: str) -> None:
        return None

    def add_chunks(self, nodes: Sequence[BaseNode]) -> None:
        return None

    def chunk_count(self) -> int:
        return 0

    def load_records(self) -> list[dict[str, Any]]:
        return []

    def load_retriever(self) -> Any | None:
        return None
