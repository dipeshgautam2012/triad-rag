"""Retrieval HTTP API — ingest files, search chunks, manage indices."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.ingest.corpus import list_corpus_files, unlink_corpus_file
from app.ingest.filename_sanitizer import sanitize_corpus_filename
from app.orchestration import (
    available_indexers,
    corpus_root,
    delete_index,
    ensure_loaded,
    index_handles,
    ingest_file,
    list_indices,
    read_indexer_mode,
    remove_source,
    search_index,
    startup,
    store_root,
    validate_index_id,
)
from app.stores.chroma_vector_store import write_index_description


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    index_id: str = Field(..., min_length=1)
    rerank: bool | None = None
    expand: bool | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    # Top k chunks to return for the query.
    chunks: list[RetrievedChunk]
    index_id: str
    candidate_count: int = 0
    # NOTE: This is used for hybrid search. Bigger pool for hybrid search or Reranker.
    candidates: list[RetrievedChunk] = Field(default_factory=list)


class IngestResponse(BaseModel):
    index_id: str
    saved_as: str
    chunks_indexed: int
    embedding_model: str | None = None
    chunker: str | None = None
    indexer: str | None = None
    ready: bool


class IndexDescription(BaseModel):
    description: str = Field(default="", max_length=500)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    startup(app)
    yield
    app.state.indices.clear()


app = FastAPI(title="Triad RAG Retrieval", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "retrieval", "status": "ok"}


@app.get("/indices")
def get_indices() -> dict[str, Any]:
    return list_indices()


@app.get("/ingest/options")
def ingest_options() -> dict[str, Any]:
    return {
        "indexers": available_indexers(),
        "default_indexer": "chroma",
        "embedding_models": settings.available_embedding_models,
        "default_embedding_model": settings.default_embedding_model,
        "chunkers": settings.available_chunkers,
        "default_chunker": settings.chunker_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "hierarchical_parent_multiplier": settings.hierarchical_parent_multiplier,
        "hierarchical_chunk_sizes": settings.hierarchical_chunk_sizes,
        "hierarchical_embed_at": settings.hierarchical_embed_at,
        "sentence_window_size": settings.sentence_window_size,
        "semantic_breakpoint_percentile": settings.semantic_breakpoint_percentile,
        "semantic_buffer_size": settings.semantic_buffer_size,
        "embedder_backend": settings.embedder_backend,
        "sparse_backend": settings.sparse_backend,
        "vector_backend": settings.vector_backend,
        "node_store_backend": settings.node_store_backend,
        "reranker_backend": settings.reranker_backend,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "rerank_candidate_multiplier": settings.rerank_candidate_multiplier,
        "search_expand": settings.search_expand,
        "hybrid_candidate_multiplier": settings.hybrid_candidate_multiplier,
    }


@app.post("/indices/{index_id}/description")
def set_description(request: Request, index_id: str, body: IndexDescription) -> dict[str, Any]:
    validate_index_id(index_id)
    try:
        md = write_index_description(index_id, body.description, store_root())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no saved index for index_id={index_id!r}") from None
    h = request.app.state.indices.get(index_id)
    if h is not None:
        h.chroma.index_metadata = dict(md)
    return {"index_id": index_id, "index_metadata": md}


@app.delete("/indices/{index_id}")
def delete_index_route(request: Request, index_id: str) -> dict[str, Any]:
    delete_index(request.app, index_id)
    return {"index_id": index_id, "deleted": True}


@app.get("/indices/{index_id}/files")
def list_corpus(index_id: str) -> dict[str, Any]:
    validate_index_id(index_id)
    return {"index_id": index_id, "files": list_corpus_files(corpus_root(), index_id)}


@app.delete("/indices/{index_id}/corpus")
def clear_corpus(request: Request, index_id: str) -> dict[str, Any]:
    validate_index_id(index_id)
    deleted = []
    for name in list_corpus_files(corpus_root(), index_id):
        remove_source(request.app, index_id, name)
        deleted.append(unlink_corpus_file(corpus_root(), index_id, name))
    return {"index_id": index_id, "deleted": deleted}


@app.delete("/indices/{index_id}/files/{filename}")
def delete_corpus_file(request: Request, index_id: str, filename: str) -> dict[str, Any]:
    validate_index_id(index_id)
    try:
        name = sanitize_corpus_filename(filename)
        remove_source(request.app, index_id, name)
        deleted = unlink_corpus_file(corpus_root(), index_id, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"no corpus file {filename!r} for index_id={index_id!r}"
        ) from None
    return {"index_id": index_id, "deleted": deleted}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: Request, body: RetrieveRequest) -> RetrieveResponse:
    h = index_handles(request.app, body.index_id)
    ensure_loaded(h)
    chunks_raw, pool_raw = search_index(
        h,
        mode=read_indexer_mode(body.index_id) or "chroma",
        query=body.query,
        top_k=body.top_k,
        rerank=body.rerank,
        reranker=request.app.state.reranker,
        expand=body.expand,
    )
    return RetrieveResponse(
        query=body.query,
        index_id=body.index_id,
        chunks=[RetrievedChunk.model_validate(c) for c in chunks_raw],
        candidate_count=len(pool_raw),
        candidates=[RetrievedChunk.model_validate(c) for c in pool_raw],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    index_id: str = Form(...),
    index_description: str | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    chunker_name: str | None = Form(default=None),
    indexer: str | None = Form(default=None),
) -> IngestResponse:
    return IngestResponse.model_validate(
        ingest_file(
            request.app,
            index_id=index_id,
            data=await file.read(),
            filename=file.filename,
            index_description=index_description,
            embedding_model=embedding_model,
            chunker_name=chunker_name,
            indexer=indexer,
        )
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the retrieval ASGI app.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-reload", action="store_true")
    args = p.parse_args(argv)
    uvicorn.run("app.main:app", host=args.host, port=8101, reload=not args.no_reload)


if __name__ == "__main__":
    main()
