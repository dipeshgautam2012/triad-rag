"""Hybrid search: merge vector and BM25 results into one list."""

from typing import Any

from llama_index.core.schema import NodeWithScore

from app.hybrid.merge_hits import merge_hybrid_hits
from app.hybrid.rank_fusion import reciprocal_rank_fusion


def combine_hybrid_results(
    vector_hits: list[NodeWithScore],
    sparse_hits: list[Any],
    *,
    limit: int,
    rank_fusion_k: int = 60,
) -> list[NodeWithScore]:
    """Rank-fuse vector + BM25 lists, then build NodeWithScore hits (up to limit)."""
    merged_ids = reciprocal_rank_fusion(
        [
            [h.node.node_id for h in vector_hits],
            [h.chunk_id for h in sparse_hits],
        ],
        k=rank_fusion_k,
    )[:limit]
    return merge_hybrid_hits(vector_hits, sparse_hits, merged_ids)
