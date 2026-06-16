"""Base class for re-ordering search hits."""

from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore


class BaseReranker(ABC):
    """Base class — second pass after retrieval; re-score hits and return top_n."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""

    @abstractmethod
    def rerank(
        self,
        hits: list[NodeWithScore],
        query: str,
        *,
        top_n: int,
    ) -> list[NodeWithScore]: ...
