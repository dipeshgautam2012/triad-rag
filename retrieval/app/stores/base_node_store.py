"""Base class for non-embedded chunks — e.g. parent passages from hierarchical chunking."""

from abc import ABC, abstractmethod
from typing import Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore


class BaseNodeStore(ABC):
    """Base class — stores parent nodes and other chunks that are not embedded.

    Hierarchical chunking embeds small leaf chunks in the vector index and keeps
    larger parents here so search can return wider context.
    """

    @abstractmethod
    def exists(self) -> bool:
        """True when this index has a persisted node store on disk."""

    @abstractmethod
    def delete_store(self) -> None:
        """Remove the persisted node store file for this index."""

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Remove nodes for one corpus file (delete or re-ingest). Other files stay."""

    @abstractmethod
    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """Append nodes and persist."""

    @abstractmethod
    def as_llama_docstore(self) -> SimpleDocumentStore:
        """LlamaIndex docstore — used to expand a leaf hit to its parent at search time."""
