from pathlib import Path
import tomllib

from pydantic_settings import BaseSettings, SettingsConfigDict

_GENERATION_ROOT = Path(__file__).resolve().parent.parent
_TOML = Path(__file__).resolve().parents[2] / "env.toml"
_PROMPTS_TOML = _GENERATION_ROOT / "prompts.toml"


def _load_section() -> dict:
    if not _TOML.is_file():
        return {}
    try:
        with _TOML.open("rb") as f:
            data = tomllib.load(f)
        g = data.get("generation")
        return g if isinstance(g, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _providers(gen: dict) -> dict[str, dict]:
    """Read [generation.google], [generation.stub], etc. — API keys and model aliases."""
    out: dict[str, dict] = {}
    for name, sec in gen.items():
        if not isinstance(sec, dict):
            continue
        models = sec.get("models")
        if not isinstance(models, dict) or not models:
            continue
        out[str(name)] = {
            "default_model_alias": str(sec.get("default_model", "")).strip(),
            "models": {str(k): str(v).strip() for k, v in models.items()},
            "api_key": str(sec.get("api_key", "")),
        }
    return out


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _load_system_prompt() -> str:
    """System instructions for the model live in generation/prompts.toml, not env.toml."""
    if not _PROMPTS_TOML.is_file():
        return ""
    try:
        with _PROMPTS_TOML.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    return str(data.get("system_prompt") or "").strip()


_GEN = _load_section()
_PROVIDERS = _providers(_GEN)


def _active_provider() -> str:
    from app.ai_providers.provider_factory import provider_implemented

    preferred = str(_GEN.get("default_provider") or "").strip().lower()
    if preferred in _PROVIDERS and provider_implemented(preferred):
        return preferred
    for name in _PROVIDERS:
        if provider_implemented(name):
            return name
    return next(iter(_PROVIDERS), "")


_provider = _active_provider()

_entry = _PROVIDERS.get(_provider, {})
_models = _entry.get("models") or {}
_alias = str(_entry.get("default_model_alias") or "") or next(iter(_models), "")


class Settings(BaseSettings):
    """Active provider name, model id, API key, temperature, and system prompt text."""

    model_config = SettingsConfigDict(env_prefix="GEN_")

    provider: str
    model_alias: str
    model: str
    api_key: str
    temperature: float
    max_output_tokens: int | None = None
    max_input_tokens: int | None = None
    reasoning_effort: str | None = None
    system_prompt: str


settings = Settings(
    provider=_provider,
    model_alias=_alias,
    model=_models.get(_alias, ""),
    api_key=str(_entry.get("api_key", "")),
    temperature=float(_GEN["temperature"]),
    max_output_tokens=_optional_int(_GEN.get("max_output_tokens")),
    max_input_tokens=_optional_int(_GEN.get("max_input_tokens")),
    reasoning_effort=(
        None
        if _GEN.get("reasoning_effort") is None
        else (str(_GEN.get("reasoning_effort")).strip() or None)
    ),
    system_prompt=_load_system_prompt(),
)


def providers() -> dict[str, dict]:
    return _PROVIDERS


def list_models() -> dict:
    from app.ai_providers.provider_factory import provider_implemented

    return {
        "provider": settings.provider,
        "model_alias": settings.model_alias,
        "model": settings.model,
        "providers": {
            name: {
                "default_model_alias": entry["default_model_alias"],
                "models": entry["models"],
            }
            for name, entry in _PROVIDERS.items()
            if provider_implemented(name)
        },
    }


def set_model(provider: str, model_alias: str) -> dict:
    entry = _PROVIDERS[provider]
    settings.provider = provider
    settings.model_alias = model_alias
    settings.model = entry["models"][model_alias]
    settings.api_key = entry["api_key"]
    return list_models()
