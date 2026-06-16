"""BM25 sparse store — chunk records in meta.json (bm25s index in same folder)."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import bm25s
from llama_index.core.schema import BaseNode, MetadataMode

from app.stores.base_sparse_store import BaseSparseStore

logger = logging.getLogger(__name__)
_META_NAME = "meta.json"


def _record_from_node(node: BaseNode) -> dict[str, Any]:
    return {
        "chunk_id": node.node_id,
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "source": str((node.metadata or {}).get("source", "")),
        "metadata": dict(node.metadata or {}),
    }


class JsonBm25SparseStore(BaseSparseStore):
    """Under store_root/sparse/<index_id>/ — meta.json + bm25s files."""

    def __init__(self, index_id: str, *, store_root: Path) -> None:
        self._index_id = index_id
        self._dir = (store_root.resolve() / "sparse" / index_id).resolve()

    @property
    def active(self) -> bool:
        return True

    def _meta_path(self) -> Path:
        return self._dir / _META_NAME

    def load_records(self) -> list[dict[str, Any]]:
        path = self._meta_path()
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data

    def _save_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            if self._dir.exists():
                shutil.rmtree(self._dir)
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path().write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rebuild(self, records: list[dict[str, Any]]) -> None:
        if not records:
            self._save_records([])
            return
        texts = [r["text"] for r in records]
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(texts))
        self._dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self._dir), corpus=texts)
        self._save_records(records)
        logger.info("Rebuilt BM25 sparse index for %s (%s chunks)", self._index_id, len(records))

    def delete_store(self) -> None:
        if self._dir.exists():
            shutil.rmtree(self._dir)

    def delete_by_source(self, source: str) -> None:
        name = (source or "").strip()
        if not name:
            return
        records = [r for r in self.load_records() if r.get("source") != name]
        # BM25 index cannot be updated in place — rewrite meta.json and rebuild
        self._rebuild(records)

    def add_chunks(self, nodes: Sequence[BaseNode]) -> None:
        if not nodes:
            return
        # Load existing records and add new nodes from input
        records = self.load_records()
        records.extend(_record_from_node(node) for node in nodes)
        # The inverted index cannot be updated in place, so rebuild from all records
        self._rebuild(records)

    def chunk_count(self) -> int:
        return len(self.load_records())

    def load_retriever(self) -> Any | None:
        if not self._meta_path().is_file():
            return None
        return bm25s.BM25.load(str(self._dir), load_corpus=False)
