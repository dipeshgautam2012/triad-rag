"""Section-based chunker — markdown only; splits on # headings, then sub-splits oversized sections.

PDF and other formats must be converted to markdown before ingest. This service does not
convert uploads; it only parses markdown syntax in the file text.
"""

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode

from app.chunkers.base_chunker import BaseChunker


class MarkdownChunker(BaseChunker):
    """Section-based chunking from markdown ``#`` headings. Chunker id: ``markdown``.

    Requires markdown source text. PDF/other formats need an upstream markdown conversion.
    """

    name = "markdown"

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        section_parser = MarkdownNodeParser.from_defaults(include_metadata=True)
        section_nodes = list(section_parser.get_nodes_from_documents(documents))
        if not section_nodes:
            return []

        size_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
        )
        chunks: list[BaseNode] = []
        for node in section_nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            if len(text) <= self.chunk_size:
                chunks.append(node)
                continue
            sub_doc = Document(text=text, metadata=dict(node.metadata or {}))
            chunks.extend(size_parser.get_nodes_from_documents([sub_doc]))
        return chunks
