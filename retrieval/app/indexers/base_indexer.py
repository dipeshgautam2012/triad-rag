"""Base class for saving chunks and searching one named index."""

import re
from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.schema import BaseNode, TextNode

from app.chunkers.base_chunker import Chunk, ChunkingResult
from app.stores.base_node_store import BaseNodeStore

_CONTEXT_NODE_ID = "context_node_id"

_INDEX_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_index_id(index_id: str) -> None:
    if not _INDEX_ID.match(index_id or ""):
        raise ValueError(
            "index_id must be 1–64 chars: letters, digits, underscore, hyphen "
            "(use 'default' for the root corpus folder)"
        )


def searchable_nodes(chunks: list[Chunk]) -> list[TextNode]:
    return [TextNode(text=c.text, metadata=c.metadata.to_dict(), id_=c.chunk_id) for c in chunks]


def context_nodes(chunks: list[Chunk]) -> list[BaseNode] | None:
    if not any(c.context for c in chunks):
        return None
    seen: set[str] = set()
    out: list[BaseNode] = []
    for chunk in chunks:
        if chunk.context is None or chunk.context.chunk_id in seen:
            continue
        seen.add(chunk.context.chunk_id)
        out.append(
            TextNode(
                text=chunk.context.text,
                metadata=chunk.context.metadata.to_dict(),
                id_=chunk.context.chunk_id,
            )
        )
    return out or None


def _expand_hit_context(
    hits: list[dict[str, Any]],
    node_store: BaseNodeStore,
) -> list[dict[str, Any]]:
    """Replace hit text from node store when metadata has context_node_id."""
    if not node_store.exists():
        return hits
    docstore = node_store.as_llama_docstore()
    out: list[dict[str, Any]] = []
    for hit in hits:
        md = dict(hit.get("metadata") or {})
        text = str(hit["text"])
        context_id = md.get(_CONTEXT_NODE_ID)
        if context_id and docstore.document_exists(context_id):
            node = docstore.get_document(context_id)
            if node is not None:
                text = node.get_content()
        out.append({**hit, "text": text})
    return out


def format_search_hit(
    chunk_id: str,
    text: str,
    score: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "metadata": dict(metadata),
    }


class BaseIndexer(ABC):
    """Minimum contract shared by vector and sparse indexers."""

    index_id: str

    def __init__(self, index_id: str) -> None:
        validate_index_id(index_id)
        self.index_id = index_id
        self._loaded = False

    @abstractmethod
    def remove_source(self, source: str) -> None:
        """Remove indexed chunks for one source filename from this indexer's store."""

    @abstractmethod
    def delete_index(self) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def add_chunks(self, result: ChunkingResult) -> int: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def chunk_count(self) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int, *, expand: bool = False) -> list[dict[str, Any]]: ...
