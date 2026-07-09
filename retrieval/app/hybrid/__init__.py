"""Hybrid search: RRF fuse vector + BM25 hits."""

from typing import Any

from llama_index.core.schema import BaseNode, NodeWithScore, TextNode


def _reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """Rank fusion (RRF): boost chunks that appear near the top in both lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)


def _node_with_score(node: BaseNode, score: float) -> NodeWithScore:
    return NodeWithScore.model_validate({"node": node, "score": score})


def node_from_retrieved(hit: dict[str, Any]) -> NodeWithScore:
    node = TextNode(
        text=hit["text"],
        metadata=hit.get("metadata") or {},
        id_=hit["chunk_id"],
    )
    return _node_with_score(node, float(hit.get("score") or 0.0))


def format_retrieved(hit: NodeWithScore) -> dict[str, Any]:
    node = hit.node
    return {
        "chunk_id": node.node_id,
        "text": node.get_content(),
        "score": float(hit.score or 0.0),
        "metadata": dict(node.metadata or {}),
    }


def _merge_hybrid_hits(
    vector_hits: list[NodeWithScore],
    sparse_hits: list[dict[str, Any]],
    merged_ids: list[str],
) -> list[NodeWithScore]:
    """Walk fused ids in order. Prefer the vector hit when both found the chunk."""
    by_vector = {h.node.node_id: h for h in vector_hits}
    by_sparse = {h["chunk_id"]: h for h in sparse_hits}
    merged: list[NodeWithScore] = []
    for chunk_id in merged_ids:
        hit = by_vector.get(chunk_id)
        if hit is not None:
            merged.append(hit)
            continue
        sparse = by_sparse.get(chunk_id)
        if sparse is None:
            continue
        node = TextNode(
            text=sparse["text"],
            metadata=sparse.get("metadata") or {},
            id_=sparse["chunk_id"],
        )
        merged.append(_node_with_score(node, float(sparse.get("score") or 0.0)))
    return merged


def combine_hybrid_results(
    vector_hits: list[NodeWithScore],
    sparse_hits: list[dict[str, Any]],
    *,
    limit: int,
    rank_fusion_k: int = 60,
) -> list[NodeWithScore]:
    """Rank-fuse vector + BM25 lists, then build NodeWithScore hits (up to limit)."""
    merged_ids = _reciprocal_rank_fusion(
        [
            [h.node.node_id for h in vector_hits],
            [h["chunk_id"] for h in sparse_hits],
        ],
        k=rank_fusion_k,
    )[:limit]
    return _merge_hybrid_hits(vector_hits, sparse_hits, merged_ids)
