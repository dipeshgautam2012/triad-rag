"""Base class and helpers for splitting corpus files into chunks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from pypdf import PdfReader

_KNOWN_METADATA_KEYS = frozenset({"source", "file_type", "chunk_role", "page"})


@dataclass
class ChunkMetadata:
    """Fields common to every chunk; chunker-specific keys live in ``extra``."""

    source: str
    file_type: str
    # chunk_role: chunk | anchor | window | parent | leaf
    chunk_role: str = "chunk"
    page: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "file_type": self.file_type,
            "chunk_role": self.chunk_role,
        }
        if self.page is not None:
            d["page"] = self.page
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChunkMetadata":
        raw = dict(data or {})
        extra = {k: v for k, v in raw.items() if k not in _KNOWN_METADATA_KEYS}
        page = raw.get("page")
        return cls(
            source=str(raw.get("source", "unknown")),
            file_type=str(raw.get("file_type", "")),
            chunk_role=str(raw.get("chunk_role", "chunk")),
            page=int(page) if page is not None else None,
            extra=extra,
        )


@dataclass
class ContextChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    context: ContextChunk | None = None


@dataclass
class ChunkingResult:
    chunker: str
    chunks: list[Chunk]


def _file_to_documents(path: Path) -> list[Document]:
    filename = path.name
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, metadata={"source": filename, "file_type": "txt"})]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        docs: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={"source": filename, "file_type": "pdf", "page": page_num},
                    )
                )
        return docs
    return []


class BaseChunker:
    """Read a file, split it, return a ChunkingResult. Override hooks for chunker-specific traits."""

    name: str = "base"

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        raise NotImplementedError

    def _file_prefix(self, md: dict) -> tuple[str, int | None, str]:
        source = str(md.get("source", "unknown"))
        page = md.get("page", None)
        page_num: int | None = int(page) if page is not None else None
        prefix = f"{source}#p{page_num}" if page_num is not None else f"{source}#"
        return source, page_num, prefix

    def _next_index(
        self,
        counts: dict[tuple[str, int | None, str], int],
        source: str,
        page: int | None,
        counter_key: str,
    ) -> int:
        key = (source, page, counter_key)
        n = counts.get(key, 0)
        counts[key] = n + 1
        return n

    def _assign_chunk_ids(self, chunks: list[BaseNode]) -> list[BaseNode]:
        """Stable ids from filename + page + chunk_role (not random UUIDs)."""
        counts: dict[tuple[str, int | None, str], int] = {}

        for chunk in chunks:
            md = dict(chunk.metadata or {})
            source, page_num, prefix = self._file_prefix(md)
            role = str(md.get("chunk_role", "chunk"))
            md.setdefault("chunk_role", role)
            i = self._next_index(counts, source, page_num, role)
            cid = f"{prefix}{role}{i}"
            chunk.metadata, chunk.node_id = md, cid
        return chunks

    def _build_chunks(self, nodes: list[BaseNode]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=node.node_id,
                text=node.get_content(),
                metadata=ChunkMetadata.from_dict(node.metadata or {}),
            )
            for node in nodes
        ]

    def _chunk_documents(self, documents: list[Document]) -> ChunkingResult:
        if not documents:
            return ChunkingResult(chunker=self.name, chunks=[])
        nodes = self._parse_documents(documents)
        nodes = self._assign_chunk_ids(nodes)
        return ChunkingResult(chunker=self.name, chunks=self._build_chunks(nodes))

    def chunk_text(
        self,
        text: str,
        source: str = "inline",
        *,
        file_type: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> ChunkingResult:
        md: dict[str, Any] = {"source": source, "file_type": file_type}
        if metadata:
            md.update(metadata)
        return self._chunk_documents([Document(text=text, metadata=md)])

    def chunk_file(self, path: Path) -> ChunkingResult:
        return self._chunk_documents(_file_to_documents(path))

    def chunk_corpus(self, corpus_dir: Path) -> ChunkingResult:
        if not corpus_dir.is_dir():
            return ChunkingResult(chunker=self.name, chunks=[])
        paths = sorted([*corpus_dir.glob("*.txt"), *corpus_dir.glob("*.pdf")], key=lambda p: p.name)
        chunks: list[Chunk] = []
        for path in paths:
            chunks.extend(self._chunk_documents(_file_to_documents(path)).chunks)
        return ChunkingResult(chunker=self.name, chunks=chunks)
