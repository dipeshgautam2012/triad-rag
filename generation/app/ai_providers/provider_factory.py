"""Pick provider class from env.toml default_provider. Called from generation main only."""

import sys
from pathlib import Path
from typing import Any

from app.ai_providers.base_provider import BaseAIProvider
from app.ai_providers.gemini_provider import GeminiProvider
from app.ai_providers.stub_provider import StubProvider

# Maps env.toml ``default_provider`` value → class. Add a new ``*_provider.py`` + one line here.
_PROVIDER_REGISTRY: dict[str, type[BaseAIProvider]] = {
    "google": GeminiProvider,
    "stub": StubProvider,
}


def provider_implemented(name: str) -> bool:
    """True if we have code for this provider name (not just a TOML block)."""
    return name.strip().lower() in _PROVIDER_REGISTRY


def make_provider(cfg: Any) -> BaseAIProvider:
    """Build a provider from a config object (``settings`` from main). Unknown names → ``StubProvider``."""
    key = cfg.provider.strip().lower()
    cls = _PROVIDER_REGISTRY.get(key, StubProvider)
    return cls(cfg)


if __name__ == "__main__":
    # Quick manual test from ``generation/``: python -m app.ai_providers.provider_factory
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from pydantic import BaseModel

    from app.config import settings

    sample_question = "What company is described in the context?"
    sample_context = "Google was founded in 1998 in Mountain View, USA. CEO is Sundar Pichai."

    provider = make_provider(settings)
    print(f"provider={settings.provider!r} model={settings.model!r} class={type(provider).__name__}")
    print("--- plain text ---")
    print(provider.generate(sample_question, sample_context))

    class CompanyFact(BaseModel):
        company_name: str
        ceo: str

    print("--- structured ---")
    print(provider.generate(sample_question, sample_context, output_schema=CompanyFact))
