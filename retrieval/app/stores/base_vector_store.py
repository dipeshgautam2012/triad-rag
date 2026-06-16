"""Base class for embedded chunks — the pieces similarity search runs over."""

from abc import ABC, abstractmethod
from typing import Any


class IndexSnapshotError(Exception):
    """Re-ingest conflicts with saved settings (e.g. different embedding model or chunker)."""


class BaseVectorStore(ABC):
    """Base class — save embedded chunks and run similarity search.

    Holds the chunks a chunker chose to embed (often hierarchical leaf chunks).
    One instance per index_id.
    """

    @abstractmethod
    def exists(self) -> bool:
        """True when this index has a persisted vector collection."""

    @abstractmethod
    def delete_store(self) -> None:
        """Remove the entire vector collection for this index."""

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Drop embedded chunks whose metadata ``source`` matches the corpus filename."""

    @abstractmethod
    def chunk_count(self) -> int:
        """Number of embedded chunks in this index."""

    @abstractmethod
    def try_get_collection(self) -> Any | None:
        """Return the backing collection, or ``None`` if it does not exist."""

    @abstractmethod
    def get_collection(self) -> Any:
        """Return the backing collection; raise if missing."""

    @abstractmethod
    def create_collection(
        self,
        *,
        embedding_model: str,
        chunker: str,
        description: str | None = None,
    ) -> Any:
        """Create the collection with ingest metadata."""

    @abstractmethod
    def resolve_embedding_model(self) -> str:
        """Read ``embedding_model`` from collection metadata."""

    @abstractmethod
    def resolve_chunker(self) -> str:
        """Read ``chunker`` from collection metadata."""

    @abstractmethod
    def read_description(self) -> str:
        """Read optional ``description`` from collection metadata."""

    @abstractmethod
    def modify_metadata(self, metadata: dict[str, Any]) -> None:
        """Replace collection metadata."""
