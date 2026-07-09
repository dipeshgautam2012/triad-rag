"""One anchor sentence per searchable node; window text in the node store."""

from llama_index.core import Document
from llama_index.core.node_parser.text.sentence_window import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode

from app.chunkers.base_chunker import (
    BaseChunker,
    Chunk,
    ChunkMetadata,
    ContextChunk,
)

_WINDOW_KEY = "window"
_ORIGINAL_TEXT_KEY = "original_text"
_CONTEXT_NODE_ID_KEY = "context_node_id"


class SentenceWindowChunker(BaseChunker):
    """One Chunk per anchor sentence; window text in context."""

    name = "sentence_window"

    def __init__(self, *, window_size: int = 3) -> None:
        self.window_size = max(1, window_size)

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size,
            window_metadata_key=_WINDOW_KEY,
            original_text_metadata_key=_ORIGINAL_TEXT_KEY,
            include_metadata=True,
        )
        nodes = list(parser.get_nodes_from_documents(documents))
        for node in nodes:
            md = dict(node.metadata or {})
            md["chunk_role"] = "anchor"
            node.metadata = md
        return nodes

    def _build_chunks(self, nodes: list[BaseNode]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in nodes:
            md = dict(node.metadata or {})
            window_text = md.pop(_WINDOW_KEY)
            md.pop(_ORIGINAL_TEXT_KEY, None)

            # the context id is the node id plus #window
            context_id = f"{node.node_id}#window"
            meta = ChunkMetadata.from_dict(md)
            meta.extra[_CONTEXT_NODE_ID_KEY] = context_id
            chunks.append(
                Chunk(
                    chunk_id=node.node_id,
                    text=node.get_content(),
                    metadata=meta,
                    context=ContextChunk(
                        chunk_id=context_id,
                        text=window_text.strip(),
                        metadata=ChunkMetadata.from_dict({**md, "chunk_role": "window"}),
                    ),
                )
            )
        return chunks
