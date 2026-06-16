"""Split where topic changes, using embedding similarity."""

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """Split where the topic shifts. Needs an embed_model (passed from make_chunker)."""

    name = "semantic"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embed_model: BaseEmbedding,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.embed_model = embed_model
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = max(1, buffer_size)

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            buffer_size=self.buffer_size,
            include_metadata=True,
        )
        return list(parser.get_nodes_from_documents(documents))
