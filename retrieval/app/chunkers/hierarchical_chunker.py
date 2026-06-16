"""Parent/child chunks — search small pieces, expand to parent when useful."""

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import BaseChunker


class HierarchicalChunker(BaseChunker):
    """Embed leaf chunks; save parent passages for auto-merge to wider context at search."""

    name = "hierarchical"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        parent_multiplier: int = 3,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.parent_multiplier = max(2, parent_multiplier)

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parent_size = self.chunk_size * self.parent_multiplier
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_size, self.chunk_size],
            chunk_overlap=self.chunk_overlap,
            include_metadata=True,
        )
        return list(parser.get_nodes_from_documents(documents))

    def _select_chunks_to_index(self, chunks: list[BaseNode]) -> list[BaseNode]:
        return get_leaf_nodes(chunks)
