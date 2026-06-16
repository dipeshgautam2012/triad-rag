"""Base class for turning a question + context into an answer."""

from typing import Any


class BaseAIProvider:
    """Base class — subclass implements generate() using cfg (model, API key, system prompt)."""

    def __init__(self, cfg: Any) -> None:
        # Runtime settings from config (model name, API key, system prompt text, etc.)
        self._cfg = cfg

    def _system_instruction(self) -> str:
        return (self._cfg.system_prompt or "").strip()

    @staticmethod
    def _user_content(question: str, context: str) -> str:
        """Format retrieved chunk text and the user question for the model."""
        q = question.strip()
        ctx = context.strip()
        ctx_block = ctx if ctx else "(none — no retrieval context was provided.)"
        return f"Context:\n{ctx_block}\n\nQuestion:\n{q}"

    def generate(
        self,
        question: str,
        context: str,
        output_schema: Any | None = None,
    ) -> Any:
        raise NotImplementedError
