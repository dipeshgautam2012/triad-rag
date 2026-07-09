"""Section-based chunker — markdown only; splits on # headings, then sub-splits oversized sections.

PDF and other formats must be converted to markdown before ingest. This service does not
convert uploads; it only parses markdown syntax in the file text.
"""

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode

from app.chunkers.base_chunker import BaseChunker

_PARAGRAPH_SEPARATOR = "\n\n"


class MarkdownChunker(BaseChunker):
    """Section-based chunking from markdown ``#`` headings. Chunker id: ``markdown``.

    Requires markdown source text. PDF/other formats need an upstream markdown conversion.
    """

    name = "markdown"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:

        # Split by markdown structure (#, ##, ###, etc.)
        section_parser = MarkdownNodeParser.from_defaults(include_metadata=True)
        section_nodes = list(section_parser.get_nodes_from_documents(documents))
        if not section_nodes:
            return []

        # Split by sentence length
        size_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=_PARAGRAPH_SEPARATOR,
        )
        chunks: list[BaseNode] = []
        # Split each section into chunks of the desired size if it's too large
        for node in section_nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            if len(text) <= self.chunk_size:
                chunks.append(node)
                continue
            sub_doc = Document(text=text, metadata=dict(node.metadata or {}))
            chunks.extend(size_parser.get_nodes_from_documents([sub_doc]))
        return chunks
