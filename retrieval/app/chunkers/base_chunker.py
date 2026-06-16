"""Base class and helpers for splitting corpus files into chunks."""

from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship
from pypdf import PdfReader

SENTENCE_WINDOW_KEY = "window"

_CHUNK_SEGMENT = "chunk"
_PARENT_SEGMENT = "parent"


def _set_chunk_node_ids(chunks: list[BaseNode], *, hierarchical: bool) -> list[BaseNode]:
    """Stable chunk ids from filename + page + position (not random UUIDs)."""
    counts: dict[tuple[str, int | None, str], int] = {}
    parent_ids: dict[str, str] = {}

    def next_index(source: str, page: int | None, counter_key: str) -> int:
        key = (source, page, counter_key)
        n = counts.get(key, 0)
        counts[key] = n + 1
        return n

    def file_prefix(md: dict) -> tuple[str, int | None, str]:
        source = str(md.get("source", "unknown"))
        page = md.get("page", None)
        page_num: int | None = int(page) if page is not None else None
        prefix = f"{source}#p{page_num}" if page_num is not None else f"{source}#"
        return source, page_num, prefix

    for chunk in chunks:
        md = dict(chunk.metadata or {})
        source, page_num, prefix = file_prefix(md)

        if not hierarchical:
            i = next_index(source, page_num, "chunk")
            cid = f"{prefix}{_CHUNK_SEGMENT}{i}"
            md.update(chunk_role="chunk", level=0)
            chunk.metadata, chunk.node_id = md, cid
            continue

        if NodeRelationship.CHILD in chunk.relationships:
            i = next_index(source, page_num, "parent")
            cid = f"{prefix}{_PARENT_SEGMENT}{i}"
            parent_ids[chunk.node_id] = cid
            md.update(chunk_role="parent", level=0)
            chunk.metadata, chunk.node_id = md, cid
            continue

        parent_ref = chunk.relationships.get(NodeRelationship.PARENT)
        if parent_ref is None:
            continue
        parent_id = parent_ids.get(parent_ref.node_id, parent_ref.node_id)
        i = next_index(source, page_num, f"child_of:{parent_id}")
        cid = f"{parent_id}{_CHUNK_SEGMENT}{i}"
        md.update(chunk_role="child", level=1, parent_id=parent_id)
        chunk.metadata, chunk.node_id = md, cid
        chunk.relationships[NodeRelationship.PARENT] = parent_ref.model_copy(
            update={"node_id": parent_id}
        )
    return chunks


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


@dataclass
class ChunkSet:
    """embed_chunks: what gets embedded. all_chunks: full set when parents are kept
    unembedded (hierarchical chunking sets this).
    """

    embed_chunks: list[BaseNode]
    all_chunks: list[BaseNode] | None = None


class BaseChunker:
    """Base class — read a file, split it, return a ChunkSet of what to embed."""

    name: str = "base"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        raise NotImplementedError

    def _assign_chunk_ids(self, chunks: list[BaseNode]) -> list[BaseNode]:
        return _set_chunk_node_ids(chunks, hierarchical=self.name == "hierarchical")

    def _select_chunks_to_index(self, chunks: list[BaseNode]) -> list[BaseNode]:
        return chunks

    def _chunk_documents(self, documents: list[Document]) -> ChunkSet:
        if not documents:
            return ChunkSet(embed_chunks=[])
        chunks = self._parse_documents(documents)
        chunks = self._assign_chunk_ids(chunks)
        embed_chunks = self._select_chunks_to_index(chunks)
        all_chunks = chunks if len(chunks) > len(embed_chunks) else None
        return ChunkSet(embed_chunks=embed_chunks, all_chunks=all_chunks)

    def chunk_file(self, path: Path) -> ChunkSet:
        return self._chunk_documents(_file_to_documents(path))

    def chunk_corpus(self, corpus_dir: Path) -> ChunkSet:
        if not corpus_dir.is_dir():
            return ChunkSet(embed_chunks=[])
        paths = sorted([*corpus_dir.glob("*.txt"), *corpus_dir.glob("*.pdf")], key=lambda p: p.name)
        all_chunks: list[BaseNode] = []
        embed_chunks: list[BaseNode] = []
        has_extras = False
        for path in paths:
            part = self.chunk_file(path)
            embed_chunks.extend(part.embed_chunks)
            all_chunks.extend(part.all_chunks if part.all_chunks is not None else part.embed_chunks)
            has_extras = has_extras or part.all_chunks is not None
        return ChunkSet(embed_chunks=embed_chunks, all_chunks=all_chunks if has_extras else None)
