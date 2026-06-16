"""Generation service: turn question + retrieved context into an answer (``POST /generate``)."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import list_models, providers, set_model, settings
from app.ai_providers.provider_factory import make_provider, provider_implemented

logger = logging.getLogger(__name__)
app = FastAPI(title="Triad RAG Generation", version="0.1.0")


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    context: str = Field(default="", max_length=200_000)


class GenerateResponse(BaseModel):
    answer: str
    model: str = settings.model


class ModelChoice(BaseModel):
    provider: str = Field(..., min_length=1)
    model_alias: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "generation", "status": "ok"}


@app.get("/models")
def get_models() -> dict:
    """List configured LLM providers/models and the active selection."""
    return list_models()


@app.post("/models/select")
def select_model(body: ModelChoice) -> dict:
    """Set the active LLM provider and model alias (runtime, in-memory)."""
    provider = body.provider.strip().lower()
    alias = body.model_alias.strip()
    catalog = providers()
    if provider not in catalog:
        raise HTTPException(status_code=400, detail=f"provider {provider!r} is not configured")
    if not provider_implemented(provider):
        raise HTTPException(status_code=400, detail=f"provider {provider!r} is not implemented yet")
    if alias not in catalog[provider]["models"]:
        raise HTTPException(status_code=400, detail=f"model_alias {alias!r} is not configured")
    return set_model(provider, alias)


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    logger.info("generate provider=%s model=%s ctx_len=%s", settings.provider, settings.model, len(body.context))
    answer = make_provider(settings).generate(body.question, body.context)
    return GenerateResponse(
        answer=answer,
        model=settings.model,
    )


def _parse_run_args(default_host: str, argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the generation ASGI app (dev server).")
    p.add_argument(
        "--host",
        default=default_host,
        help=f"bind host (default: {default_host})",
    )
    p.add_argument(
        "--no-reload",
        action="store_true",
        help="disable autoreload (default: reload on)",
    )
    return p.parse_args(argv)


def run(host: str, port: int, *, reload: bool = True) -> None:
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


def main(argv: list[str] | None = None) -> None:
    bind_host = "127.0.0.1"
    port = 8102
    args = _parse_run_args(bind_host, argv)
    run(args.host, port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
