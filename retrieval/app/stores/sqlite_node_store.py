"""Parent chunks saved in SQLite — one file per index."""

import json
import sqlite3
from pathlib import Path
from typing import Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc

from app.stores.base_node_store import BaseNodeStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source);
"""


class SqliteNodeStore(BaseNodeStore):
    """SQLite file per index at node_store/{index_id}.sqlite — parent chunks from hierarchical."""

    def __init__(self, index_id: str, *, store_root: Path) -> None:
        self._index_id = index_id
        self._store_root = store_root.resolve()

    def _path(self) -> Path:
        return self._store_root / "node_store" / f"{self._index_id}.sqlite"

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        return conn

    @staticmethod
    def _source(node: BaseNode) -> str:
        return str((node.metadata or {}).get("source", ""))

    def _load(self) -> SimpleDocumentStore:
        store = SimpleDocumentStore()
        if not self.exists():
            return store
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM nodes").fetchall()
        if rows:
            nodes = [json_to_doc(json.loads(row[0])) for row in rows]
            store.add_documents(nodes)
        return store

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
        with self._connect() as conn:
            conn.execute("DELETE FROM nodes WHERE source = ?", (source,))
            conn.commit()

    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        if not nodes:
            return
        # Upsert by node id; caller removes this source's rows before re-ingest
        rows = [
            (node.node_id, self._source(node), json.dumps(doc_to_json(node)))
            for node in nodes
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO nodes (node_id, source, payload) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()

    def as_llama_docstore(self) -> SimpleDocumentStore:
        return self._load()
