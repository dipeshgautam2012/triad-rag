"""Pick a chunker by name. Semantic chunker requires embed_model."""

from llama_index.core.embeddings import BaseEmbedding

from app.chunkers.base_chunker import BaseChunker
from app.chunkers.hierarchical_chunker import EmbedAt, HierarchicalChunker
from app.chunkers.markdown_chunker import MarkdownChunker
from app.chunkers.semantic_chunker import SemanticChunker
from app.chunkers.sentence_window_chunker import SentenceWindowChunker
from app.chunkers.simple_chunker import SimpleChunker

CHUNKERS = frozenset({"simple", "hierarchical", "markdown", "sentence_window", "semantic"})


def make_chunker(
    name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: BaseEmbedding | None = None,
    hierarchical_parent_multiplier: int = 3,
    hierarchical_chunk_sizes: list[int] | None = None,
    hierarchical_embed_at: EmbedAt = "leaves",
    sentence_window_size: int = 3,
    semantic_breakpoint_percentile: int = 95,
    semantic_buffer_size: int = 1,
) -> BaseChunker:
    key = name.strip().lower()
    if key not in CHUNKERS:
        raise ValueError(f"unsupported chunker: {name!r}; supported: {sorted(CHUNKERS)}")
    if key == "hierarchical":
        return HierarchicalChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_sizes=hierarchical_chunk_sizes,
            parent_multiplier=hierarchical_parent_multiplier,
            embed_at=hierarchical_embed_at,
        )
    if key == "markdown":
        return MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if key == "sentence_window":
        return SentenceWindowChunker(window_size=sentence_window_size)
    if key == "semantic":
        if embed_model is None:
            raise ValueError("semantic chunker requires embed_model")
        return SemanticChunker(
            embed_model=embed_model,
            breakpoint_percentile_threshold=semantic_breakpoint_percentile,
            buffer_size=semantic_buffer_size,
        )
    return SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
