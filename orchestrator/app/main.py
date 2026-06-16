"""Orchestrator: POST /query → retrieval → generation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

app = FastAPI(title="Triad RAG Orchestrator", version="0.1.0")

_RETRIEVAL = settings.retrieval_url.rstrip("/")
_GENERATION = settings.generation_url.rstrip("/")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    index_id: str = Field(..., min_length=1, description="Logical retrieval index.")
    rerank: bool | None = Field(
        default=None,
        description="Pass through to retrieval POST /retrieve. None = retrieval server default (rerank_enabled in env.toml).",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IndexDescription(BaseModel):
    description: str = Field(default="", max_length=500)


class ModelChoice(BaseModel):
    provider: str = Field(..., min_length=1)
    model_alias: str = Field(..., min_length=1)


def _chunks_to_context(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks):
        block = f"[{i + 1}] {c.get('text', '')}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip() or "(empty context)"

# Retryable status codes: 502, 503, 504
_RETRYABLE_STATUS = frozenset({502, 503, 504})


async def _call_service(method: str, url: str, **kwargs: Any) -> Any:
    """HTTP call to retrieval or generation; returns parsed JSON body."""
    attempts = settings.retry_attempts
    async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(settings.retry_wait_s * attempt)
            try:
                r = await client.request(method, url, **kwargs)
            except httpx.RequestError as e:
                if attempt + 1 < attempts:
                    continue
                raise HTTPException(status_code=502, detail=f"Downstream unavailable: {e!s}") from e
            if r.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
                continue
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text[:500])
            return r.json()
    raise HTTPException(status_code=502, detail="Downstream unavailable after retries")


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "orchestrator", "status": "ok"}


@app.get("/indices")
async def list_indices() -> dict[str, Any]:
    return await _call_service("GET", f"{_RETRIEVAL}/indices")


@app.get("/models")
async def get_models() -> dict[str, Any]:
    return await _call_service("GET", f"{_GENERATION}/models")


@app.post("/models/select")
async def select_model(body: ModelChoice) -> dict[str, Any]:
    return await _call_service("POST", f"{_GENERATION}/models/select", json=body.model_dump())


@app.post("/indices/{index_id}/description")
async def set_index_description(index_id: str, body: IndexDescription) -> dict[str, Any]:
    return await _call_service(
        "POST",
        f"{_RETRIEVAL}/indices/{index_id}/description",
        json=body.model_dump(),
    )


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    q = body.question.strip()
    payload = await _call_service(
        "POST",
        f"{_RETRIEVAL}/retrieve",
        json={
            "query": q,
            "top_k": int(body.top_k),
            "index_id": body.index_id,
            "rerank": body.rerank,
        },
    )
    raw_chunks = payload.get("chunks") or []
    sources = [
        SourceChunk(
            chunk_id=str(c.get("chunk_id", "")),
            text=str(c.get("text", "")),
            score=c.get("score"),
            metadata=dict(c.get("metadata") or {}),
        )
        for c in raw_chunks
    ]
    if not sources:
        return QueryResponse(
            answer="No passages retrieved (empty corpus or no good hits).",
            sources=[],
        )

    context = _chunks_to_context([s.model_dump() for s in sources])
    gen_payload = await _call_service(
        "POST",
        f"{_GENERATION}/generate",
        json={"question": q, "context": context},
    )
    answer = str(gen_payload.get("answer", "")).strip() or "[empty answer]"
    return QueryResponse(answer=answer, sources=sources)


def _parse_run_args(default_host: str, argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the orchestrator ASGI app (dev server).")
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
    port = 8100
    args = _parse_run_args(bind_host, argv)
    run(args.host, port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
