"""Base class for text embedding models."""

from abc import ABC, abstractmethod

from llama_index.core.embeddings import BaseEmbedding


class BaseEmbedder(ABC):
    """Base class — wraps one embedding model for ingest, search, and semantic chunking."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""

    @property
    @abstractmethod
    def embedding_model(self) -> BaseEmbedding:
        """LlamaIndex embedding object for this model."""
