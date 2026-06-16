"""sentence-transformers model via LlamaIndex HuggingFaceEmbedding."""

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.embedders.base_embedder import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """Default embedder — one sentence-transformers model (e.g. all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = HuggingFaceEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_model(self) -> BaseEmbedding:
        return self._model
