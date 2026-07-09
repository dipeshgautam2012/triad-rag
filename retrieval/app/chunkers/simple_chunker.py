"""Fixed-size text splits with overlap."""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import BaseChunker

_PARAGRAPH_SEPARATOR = "\n\n"


class SimpleChunker(BaseChunker):
    """Default chunker — fixed-size splits with overlap."""

    name = "simple"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=_PARAGRAPH_SEPARATOR,
        )
        return list(parser.get_nodes_from_documents(documents))
