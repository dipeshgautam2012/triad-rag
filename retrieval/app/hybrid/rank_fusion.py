"""Merge two ranked chunk lists into one combined ranking."""


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """Rank fusion (RRF): boost chunks that appear near the top in both vector and BM25 lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
