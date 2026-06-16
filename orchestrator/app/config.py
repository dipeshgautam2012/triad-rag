from pathlib import Path
import tomllib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_TOML = Path(__file__).resolve().parents[2] / "env.toml"


def _load_section() -> dict:
    if not _TOML.is_file():
        return {}
    try:
        with _TOML.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    section = data.get("orchestrator")
    return section if isinstance(section, dict) else {}


class Settings(BaseSettings):
    """Fields match [orchestrator] in env.toml — no duplicate defaults here."""

    model_config = SettingsConfigDict(env_prefix="ORCH_")

    retrieval_url: str
    generation_url: str
    request_timeout_s: float
    retry_attempts: int = Field(ge=1)
    retry_wait_s: float = Field(ge=0)


settings = Settings(**_load_section())
