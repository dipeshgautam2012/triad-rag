"""Build an embedder from model_name and backend passed by main. Cached per model name."""

from app.embedders.base_embedder import BaseEmbedder
from app.embedders.huggingface_embedder import HuggingFaceEmbedder

EMBEDDER_BACKENDS = frozenset({"huggingface"})

_embedders: dict[str, BaseEmbedder] = {}


def make_embedder(model_name: str, *, backend: str) -> BaseEmbedder:
    name = (model_name or "").strip()
    if not name:
        raise ValueError("embedding model name is required")
    key = backend.strip().lower()
    if key not in EMBEDDER_BACKENDS:
        raise ValueError(
            f"unsupported embedder backend: {backend!r}; supported: {sorted(EMBEDDER_BACKENDS)}"
        )
    cached = _embedders.get(name)
    if cached is not None:
        return cached
    embedder = HuggingFaceEmbedder(name)
    _embedders[name] = embedder
    return embedder
