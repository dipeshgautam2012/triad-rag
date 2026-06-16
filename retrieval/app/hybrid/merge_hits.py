"""Turn fused chunk ids into a single ordered hit list."""

from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from app.stores.base_sparse_store import SparseHit


def node_from_retrieved(hit: dict[str, Any]) -> NodeWithScore:
    node = TextNode(
        text=hit["text"],
        metadata=hit.get("metadata") or {},
        id_=hit["chunk_id"],
    )
    return NodeWithScore(node=node, score=float(hit.get("score") or 0.0))


def sparse_hit_from_retrieved(hit: dict[str, Any]) -> SparseHit:
    return SparseHit(
        chunk_id=hit["chunk_id"],
        text=hit["text"],
        score=float(hit.get("score") or 0.0),
        metadata=dict(hit.get("metadata") or {}),
    )


def format_retrieved(hit: NodeWithScore) -> dict[str, Any]:
    node = hit.node
    return {
        "chunk_id": node.node_id,
        "text": node.get_content(),
        "score": float(hit.score or 0.0),
        "metadata": dict(node.metadata or {}),
    }


def merge_hybrid_hits(
    vector_hits: list[NodeWithScore],
    sparse_hits: list[Any],
    merged_ids: list[str],
) -> list[NodeWithScore]:
    """Walk fused ids in order. Prefer the vector hit when both found the chunk.

    Sparse hits need chunk_id, text, score, and metadata (see SparseHit).
    """
    by_vector = {h.node.node_id: h for h in vector_hits}
    by_sparse = {h.chunk_id: h for h in sparse_hits}
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
            text=sparse.text,
            metadata=sparse.metadata,
            id_=sparse.chunk_id,
        )
        merged.append(NodeWithScore(node=node, score=sparse.score))
    return merged
