"""Fake answers for local testing — no API call, no real intelligence."""

from typing import Any

from app.ai_providers.base_provider import BaseAIProvider


class StubProvider(BaseAIProvider):
    """Return a made-up reply that quotes the question and a bit of context.

    Use when ``default_provider = "stub"`` in env.toml, or as a fallback when the
    configured provider name is unknown. Good for wiring up ingest/retrieve/chat
    without spending API credits.
    """

    def generate(
        self,
        question: str,
        context: str,
        output_schema: Any | None = None,
    ) -> Any:
        q = question.strip()[:200]
        ctx = context.strip()
        ctx_preview = ctx[:300] if ctx else ""
        text = (
            f"[stub] Based on retrieved context: {ctx_preview!r}. Question: {q!r}"
            if ctx_preview
            else f"[stub] No retrieval context provided. Question: {q!r}"
        )
        if output_schema is None:
            return text
        return {"answer": text}
