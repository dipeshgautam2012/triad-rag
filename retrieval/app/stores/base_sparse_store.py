"""Base class for keyword (BM25) search over the same embedded chunks."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from llama_index.core.schema import BaseNode


class BaseSparseStore(ABC):
    """Base class — keyword search by matching query words.

    Indexes the same chunk texts as the vector store. Combined with vector search
    when hybrid mode is enabled.
    """

    @property
    @abstractmethod
    def active(self) -> bool:
        """False when keyword search is disabled (none backend)."""

    @abstractmethod
    def delete_store(self) -> None:
        """Remove the persisted sparse index for this ``index_id``."""

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Remove chunks for one corpus file (``metadata['source']``)."""

    @abstractmethod
    def add_chunks(self, nodes: Sequence[BaseNode]) -> None:
        """Index searchable chunk texts."""

    @abstractmethod
    def chunk_count(self) -> int:
        """Number of indexed chunks."""

    @abstractmethod
    def load_records(self) -> list[dict[str, Any]]:
        """Persisted chunk records (chunk_id, text, source, metadata) in index order."""

    @abstractmethod
    def load_retriever(self) -> Any | None:
        """Load the persisted BM25 retriever, or None when no index exists."""
