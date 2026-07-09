from pathlib import Path
from typing import Literal
import tomllib

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_TOML = Path(__file__).resolve().parents[2] / "env.toml"
# Resolve relative paths from retrieval/ root, not the shell cwd.
_RETRIEVAL_ROOT = Path(__file__).resolve().parent.parent


def _load_section() -> dict:
    if not _TOML.is_file():
        return {}
    try:
        with _TOML.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    section = data.get("retrieval")
    if not isinstance(section, dict):
        return {}
    if "search_expand" not in section and "hierarchical_expand_parent" in section:
        section = {**section, "search_expand": section["hierarchical_expand_parent"]}
    return section


class Settings(BaseSettings):
    """Reads the [retrieval] section of env.toml."""

    model_config = SettingsConfigDict(env_prefix="RET_")

    corpus_dir: Path
    index_store_dir: Path
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    chunker_name: str
    available_chunkers: list[str]
    hierarchical_parent_multiplier: int = Field(ge=2, le=16)
    hierarchical_chunk_sizes: list[int] | None = None
    hierarchical_embed_at: int | Literal["leaves"] = "leaves"
    search_expand: bool
    sentence_window_size: int = Field(ge=1, le=20)
    semantic_breakpoint_percentile: int = Field(ge=0, le=100)
    semantic_buffer_size: int = Field(ge=1, le=10)
    available_embedding_models: list[str]
    default_embedding_model: str
    embedding_batch_size: int = Field(ge=1)
    max_upload_bytes: int = Field(ge=1024)
    rerank_enabled: bool
    rerank_model: str
    rerank_candidate_multiplier: int = Field(ge=1)
    vector_backend: str
    node_store_backend: str
    embedder_backend: str
    reranker_backend: str
    sparse_backend: str
    hybrid_enabled: bool
    hybrid_candidate_multiplier: int = Field(ge=1)

    @field_validator("corpus_dir", "index_store_dir", mode="after")
    @classmethod
    def _resolve_data_dirs(cls, v: Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = (_RETRIEVAL_ROOT / p).resolve()
        return p.resolve()

    @model_validator(mode="after")
    def _check_lists(self) -> "Settings":
        if self.default_embedding_model not in self.available_embedding_models:
            raise ValueError(
                f"default_embedding_model {self.default_embedding_model!r} is not in "
                f"available_embedding_models {self.available_embedding_models!r}"
            )
        if self.chunker_name not in self.available_chunkers:
            raise ValueError(
                f"chunker_name {self.chunker_name!r} is not in "
                f"available_chunkers {self.available_chunkers!r}"
            )
        return self


settings = Settings(**_load_section())
