"""Fixed-size text splits with overlap."""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import BaseChunker


class SimpleChunker(BaseChunker):
    """Default chunker — fixed-size splits with overlap."""

    name = "simple"

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
        )
        return list(parser.get_nodes_from_documents(documents))
