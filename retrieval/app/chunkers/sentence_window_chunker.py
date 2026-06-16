"""One sentence per chunk with neighboring sentences for context."""

from llama_index.core import Document
from llama_index.core.node_parser.text.sentence_window import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import SENTENCE_WINDOW_KEY, BaseChunker

_ORIGINAL_TEXT_KEY = "original_text"


class SentenceWindowChunker(BaseChunker):
    """One sentence per chunk; embed a window of nearby sentences for richer matches."""

    name = "sentence_window"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        window_size: int = 3,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.window_size = max(1, window_size)

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size,
            window_metadata_key=SENTENCE_WINDOW_KEY,
            original_text_metadata_key=_ORIGINAL_TEXT_KEY,
            include_metadata=True,
        )
        nodes = list(parser.get_nodes_from_documents(documents))
        for node in nodes:
            window = (node.metadata or {}).get(SENTENCE_WINDOW_KEY)
            if isinstance(window, str) and window.strip():
                node.set_content(window)
        return nodes
