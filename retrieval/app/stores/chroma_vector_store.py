"""Chroma storage — one collection per index_id."""

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from app.stores.base_vector_store import IndexSnapshotError, BaseVectorStore

_INDEX_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _chroma_client(store_root: Path) -> chromadb.PersistentClient:
    root = store_root.resolve() / "chroma"
    root.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(root))


def _collection_metadata(
    embedding_model: str,
    chunker: str,
    description: str | None = None,
) -> dict[str, Any]:
    md: dict[str, Any] = {"embedding_model": embedding_model, "chunker": chunker}
    if description:
        md["description"] = description[:500]
    return md


def list_indices_detailed(store_root: Path) -> list[dict[str, Any]]:
    """List indices with description, embedding_model, chunker, and chunk count."""
    rows: list[dict[str, Any]] = []
    for c in _chroma_client(store_root).list_collections():
        name = getattr(c, "name", "")
        if not _INDEX_ID.match(name or ""):
            continue
        md = dict(c.metadata or {})
        try:
            chunks = int(c.count())
        except NotFoundError:
            chunks = 0
        rows.append(
            {
                "index_id": name,
                "description": str(md.get("description", "")).strip(),
                "embedding_model": str(md.get("embedding_model", "")).strip(),
                "chunker": str(md.get("chunker", "")).strip(),
                "chunks": chunks,
            }
        )
    return sorted(rows, key=lambda r: r["index_id"])


def write_index_description(
    index_id: str, description: str, store_root: Path
) -> dict[str, Any]:
    store = ChromaVectorStore(index_id, store_root=store_root)
    if not store.exists():
        raise FileNotFoundError(index_id)
    col = store.get_collection()
    md = dict(col.metadata or {})
    cleaned = description.strip()
    if cleaned:
        md["description"] = cleaned[:500]
    else:
        md.pop("description", None)
    store.modify_metadata(md)
    return md


class ChromaVectorStore(BaseVectorStore):
    """One Chroma collection per index_id. Collection metadata pins embedding_model and chunker."""

    def __init__(self, index_id: str, *, store_root: Path) -> None:
        self._index_id = index_id
        self._store_root = store_root.resolve()

    def _client(self) -> chromadb.PersistentClient:
        return _chroma_client(self._store_root)

    def exists(self) -> bool:
        return any(
            getattr(c, "name", None) == self._index_id
            for c in self._client().list_collections()
        )

    def delete_store(self) -> None:
        try:
            self._client().delete_collection(name=self._index_id)
        except NotFoundError:
            pass

    def delete_by_source(self, source: str) -> None:
        col = self.try_get_collection()
        if col is not None:
            # Chroma deletes matching rows by metadata — no full collection rebuild
            col.delete(where={"source": source})

    def chunk_count(self) -> int:
        col = self.try_get_collection()
        if col is None:
            return 0
        try:
            return int(col.count())
        except NotFoundError:
            return 0

    def try_get_collection(self) -> Any | None:
        try:
            return self._client().get_collection(name=self._index_id)
        except NotFoundError:
            return None

    def get_collection(self) -> Any:
        return self._client().get_collection(name=self._index_id)

    def create_collection(
        self,
        *,
        embedding_model: str,
        chunker: str,
        description: str | None = None,
    ) -> Any:
        return self._client().get_or_create_collection(
            name=self._index_id,
            metadata=_collection_metadata(embedding_model, chunker, description),
        )

    def resolve_embedding_model(self) -> str:
        col = self.get_collection()
        model = str((col.metadata or {}).get("embedding_model", "")).strip()
        if not model:
            raise IndexSnapshotError(
                f"index {getattr(col, 'name', '?')!r} has no recorded embedding_model"
            )
        return model

    def resolve_chunker(self) -> str:
        col = self.get_collection()
        chunker = str((col.metadata or {}).get("chunker", "")).strip()
        if not chunker:
            raise IndexSnapshotError(
                f"index {getattr(col, 'name', '?')!r} has no recorded chunker"
            )
        return chunker

    def read_description(self) -> str:
        col = self.get_collection()
        return str((col.metadata or {}).get("description", "")).strip()

    def modify_metadata(self, metadata: dict[str, Any]) -> None:
        self.get_collection().modify(metadata=metadata)
