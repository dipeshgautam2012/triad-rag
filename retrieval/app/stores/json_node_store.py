"""Parent chunks saved as one JSON file per index."""

from pathlib import Path
from typing import Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.stores.base_node_store import BaseNodeStore


class JsonNodeStore(BaseNodeStore):
    """JSON file per index at node_store/{index_id}.json — parent chunks from hierarchical."""

    def __init__(self, index_id: str, *, store_root: Path) -> None:
        self._index_id = index_id
        self._store_root = store_root.resolve()

    def _path(self) -> Path:
        return self._store_root / "node_store" / f"{self._index_id}.json"

    def _load(self) -> SimpleDocumentStore:
        path = self._path()
        if path.is_file():
            return SimpleDocumentStore.from_persist_path(str(path))
        return SimpleDocumentStore()

    def _save(self, store: SimpleDocumentStore) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        store.persist(persist_path=str(path))

    def exists(self) -> bool:
        return self._path().is_file()

    def delete_store(self) -> None:
        path = self._path()
        if path.is_file():
            path.unlink()
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def delete_by_source(self, source: str) -> None:
        """Remove nodes for one corpus file (delete or re-ingest). Other files stay."""
        if not self.exists():
            return
        store = self._load()
        for nid, node in list(store.docs.items()):
            if str((node.metadata or {}).get("source", "")) == source:
                store.delete_document(nid, raise_error=False)
        self._save(store)

    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        # Load persisted nodes, append new ones, save back (parents for hierarchical merge)
        store = self._load()
        store.add_documents(nodes)
        self._save(store)

    def as_llama_docstore(self) -> SimpleDocumentStore:
        return self._load()
