"""BM25 sparse store — chunk records in SQLite (bm25s index in same folder)."""

import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import bm25s
from llama_index.core.schema import BaseNode, MetadataMode

from app.stores.base_sparse_store import BaseSparseStore

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


def _record_from_node(node: BaseNode) -> dict[str, Any]:
    return {
        "chunk_id": node.node_id,
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "source": str((node.metadata or {}).get("source", "")),
        "metadata": dict(node.metadata or {}),
    }


class SqliteBm25SparseStore(BaseSparseStore):
    """Under store_root/sparse/<index_id>/ — chunks.sqlite + bm25s files."""

    def __init__(self, index_id: str, *, store_root: Path) -> None:
        self._index_id = index_id
        self._dir = (store_root.resolve() / "sparse" / index_id).resolve()

    @property
    def active(self) -> bool:
        return True

    def _db_path(self) -> Path:
        return self._dir / "chunks.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self._dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path())
        conn.executescript(_SCHEMA)
        return conn

    def load_records(self) -> list[dict[str, Any]]:
        if not self._db_path().is_file():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text, source, metadata FROM chunks ORDER BY rowid"
            ).fetchall()
        records: list[dict[str, Any]] = []
        for chunk_id, text, source, metadata_json in rows:
            md = json.loads(metadata_json)
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "metadata": md if isinstance(md, dict) else {},
                }
            )
        return records

    def _save_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            if self._dir.exists():
                shutil.rmtree(self._dir)
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks (chunk_id, source, text, metadata) VALUES (?, ?, ?, ?)",
                [
                    (
                        r["chunk_id"],
                        r["source"],
                        r["text"],
                        json.dumps(r.get("metadata") or {}, ensure_ascii=False),
                    )
                    for r in records
                ],
            )
            conn.commit()

    def _rebuild_bm25(self, records: list[dict[str, Any]]) -> None:
        texts = [r["text"] for r in records]
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(texts))
        self._dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self._dir), corpus=texts)
        logger.info("Rebuilt BM25 sparse index for %s (%s chunks)", self._index_id, len(records))

    def _rebuild(self, records: list[dict[str, Any]]) -> None:
        if not records:
            self._save_records([])
            return
        self._save_records(records)
        self._rebuild_bm25(records)

    def delete_store(self) -> None:
        if self._dir.exists():
            shutil.rmtree(self._dir)

    def delete_by_source(self, source: str) -> None:
        name = (source or "").strip()
        if not name:
            return
        if not self._db_path().is_file():
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (name,))
            conn.commit()
        records = self.load_records()
        if records:
            # SQLite rows already updated above; BM25 still needs a full rebuild
            self._rebuild_bm25(records)
        elif self._dir.exists():
            shutil.rmtree(self._dir)

    def add_chunks(self, nodes: Sequence[BaseNode]) -> None:
        if not nodes:
            return
        # Load existing records and add new nodes from input
        records = self.load_records()
        records.extend(_record_from_node(node) for node in nodes)
        # The inverted index of the BM25 sparse store cannot be updated in place,
        # so we need to rebuild it with the updated records
        self._rebuild(records)

    def chunk_count(self) -> int:
        if not self._db_path().is_file():
            return 0
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def load_retriever(self) -> Any | None:
        if not self._db_path().is_file():
            return None
        return bm25s.BM25.load(str(self._dir), load_corpus=False)
