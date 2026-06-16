"""Base class for saving chunks and searching one named index."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.chunkers.base_chunker import ChunkSet

_INDEX_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_index_id(index_id: str) -> None:
    if not _INDEX_ID.match(index_id or ""):
        raise ValueError(
            "index_id must be 1–64 chars: letters, digits, underscore, hyphen "
            "(use 'default' for the root corpus folder)"
        )


@dataclass(frozen=True)
class IndexerDeps:
    """Paths and search flags from main (hybrid, rerank, hierarchical parent expansion)."""

    corpus_dir: Path
    index_store_dir: Path
    default_embedding_model: str
    default_chunker_name: str
    rerank_enabled: bool
    rerank_candidate_multiplier: int
    hierarchical_expand_parent: bool
    hybrid_enabled: bool
    hybrid_candidate_multiplier: int


class BaseIndexer(ABC):
    """Base class — ingest files, delete corpus/index, and search one index_id."""

    index_id: str
    index_metadata: dict[str, Any]
    embedding_model: str | None
    chunker: str | None

    @abstractmethod
    def corpus_dir(self) -> Path: ...

    @abstractmethod
    def delete_index(self) -> None: ...

    @abstractmethod
    def list_corpus_files(self) -> list[str]: ...

    @abstractmethod
    def delete_corpus_file(self, filename: str) -> str: ...

    @abstractmethod
    def clear_corpus(self) -> list[str]: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def bind_embedder(self, embedder: Any) -> None: ...

    @abstractmethod
    def bind_reranker(self, reranker: Any) -> None: ...

    @abstractmethod
    def add_chunks(
        self,
        chunks: ChunkSet,
        *,
        source: str,
        embedder: Any,
        description: str | None = None,
        chunker_name: str | None = None,
    ) -> int: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def chunk_count(self) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int, *, rerank: bool | None = None) -> list[dict[str, Any]]: ...
