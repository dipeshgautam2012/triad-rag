"""Build a reranker from model_name and backend passed by main. Cached per model name."""

from app.rerankers.base_reranker import BaseReranker
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker

RERANKER_BACKENDS = frozenset({"cross_encoder"})

_rerankers: dict[str, BaseReranker] = {}


def make_reranker(model_name: str, *, backend: str) -> BaseReranker:
    name = (model_name or "").strip()
    if not name:
        raise ValueError("reranker model name is required")
    key = backend.strip().lower()
    if key not in RERANKER_BACKENDS:
        raise ValueError(
            f"unsupported reranker backend: {backend!r}; supported: {sorted(RERANKER_BACKENDS)}"
        )
    cached = _rerankers.get(name)
    if cached is not None:
        return cached
    reranker = CrossEncoderReranker(name)
    _rerankers[name] = reranker
    return reranker
