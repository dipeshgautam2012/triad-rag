"""Second-pass scorer using a cross-encoder model."""

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from app.rerankers.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Scores each query+chunk pair and returns the top_n most relevant hits."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._by_top_n: dict[int, SentenceTransformerRerank] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        hits: list[NodeWithScore],
        query: str,
        *,
        top_n: int,
    ) -> list[NodeWithScore]:
        n = max(1, int(top_n))
        postprocessor = self._by_top_n.get(n)
        if postprocessor is None:
            postprocessor = SentenceTransformerRerank(model=self._model_name, top_n=n)
            self._by_top_n[n] = postprocessor
        return postprocessor.postprocess_nodes(hits, query_str=query)
