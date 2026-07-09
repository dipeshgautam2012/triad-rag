# Retrieval service — API surface

> **Diagram source of truth** for retrieval service layout. HTTP in `main.py`; ingest/search wiring in `orchestration.py`. Indexer ids: `chroma` / `bm25` / `hybrid` (`vector` is a legacy alias). Parent expansion: `search_expand` (request override: `expand`). See also [`DESIGN.md`](../../DESIGN.md).

Structural reference for diagram generation: **packages, modules, classes, function prototypes, and who calls whom**. No implementations.

Root: `retrieval/app/`

---

## config

### `app.config`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `settings` | instance | `Settings` — loaded from `env.toml` `[retrieval]` |
| `Settings` | class | Pydantic settings (paths, backends, allowlists, chunk/rerank/hybrid flags) |

**`settings` fields used elsewhere:** `corpus_dir`, `index_store_dir`, `chunk_size`, `chunk_overlap`, `chunker_name`, `available_chunkers`, `hierarchical_parent_multiplier`, `hierarchical_chunk_sizes`, `hierarchical_embed_at`, `sentence_window_size`, `semantic_breakpoint_percentile`, `semantic_buffer_size`, `available_embedding_models`, `default_embedding_model`, `max_upload_bytes`, `rerank_enabled`, `rerank_model`, `rerank_candidate_multiplier`, `search_expand`, `vector_backend`, `node_store_backend`, `embedder_backend`, `reranker_backend`, `sparse_backend`, `hybrid_candidate_multiplier`

**Called from:** `app.main`, `app.orchestration` (everywhere settings are read)

---

## main (HTTP entry)

### `app.main`

#### Types / models

| Symbol | Kind | Notes |
|--------|------|-------|
| `RetrieveRequest` | Pydantic model | `query`, `top_k`, `index_id`, `rerank`, `expand` (optional; default from `settings`) |
| `RetrievedChunk` | Pydantic model | `chunk_id`, `text`, `score`, `metadata` |
| `RetrieveResponse` | Pydantic model | `query`, `chunks`, `index_id`, `candidate_count`, `candidates` |
| `IngestResponse` | Pydantic model | `index_id`, `saved_as`, `chunks_indexed`, `embedding_model`, `chunker`, `indexer`, `ready` |
| `IndexDescription` | Pydantic model | `description` |
| `app` | FastAPI | ASGI app with `lifespan` |

#### Lifespan

| Function | Prototype | Calls |
|----------|-----------|-------|
| `lifespan` | `(app: FastAPI) -> AsyncIterator[None]` | `startup(app)`; clears `app.state.indices` on shutdown |

#### HTTP routes

| Route | Handler | Calls |
|-------|---------|-------|
| `GET /health` | `health()` | — |
| `GET /indices` | `get_indices()` | `list_indices()` |
| `GET /ingest/options` | `ingest_options()` | `available_indexers()`, reads `settings` allowlists/defaults/backends/tuning |
| `POST /indices/{index_id}/description` | `set_description(...)` | `validate_index_id`, `write_index_description(...)`, updates cached `handles.chroma.index_metadata` |
| `DELETE /indices/{index_id}` | `delete_index_route(...)` | `delete_index(app, index_id)` |
| `GET /indices/{index_id}/files` | `list_corpus(...)` | `validate_index_id`, `list_corpus_files(corpus_root(), index_id)` |
| `DELETE /indices/{index_id}/corpus` | `clear_corpus(...)` | `validate_index_id`, `remove_source` per file, `unlink_corpus_file` |
| `DELETE /indices/{index_id}/files/{filename}` | `delete_corpus_file(...)` | `validate_index_id`, `sanitize_corpus_filename`, `remove_source`, `unlink_corpus_file` |
| `POST /retrieve` | `retrieve(...)` | `index_handles`, `ensure_loaded`, `read_indexer_mode`, `search_index(..., reranker=app.state.reranker, expand=body.expand)` |
| `POST /ingest` | `ingest(...)` | `ingest_file(app, ...)` |

#### CLI

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `main` | `(argv=None) -> None` | `__main__` — `uvicorn.run("app.main:app", port=8101)` |

---

## orchestration (ingest / search wiring)

### `app.orchestration`

#### Types / constants

| Symbol | Kind | Notes |
|--------|------|-------|
| `INDEXER_MODES` | constant | `("chroma", "bm25", "hybrid")` |
| `IndexHandles` | dataclass | `chroma: ChromaIndexer`, `bm25: Bm25Indexer` |

#### Path helpers

| Function | Prototype | Calls |
|----------|-----------|-------|
| `store_root` | `() -> Path` | `settings.index_store_dir` |
| `corpus_root` | `() -> Path` | `settings.corpus_dir` |
| `validate_index_id` | `(index_id: str) -> None` | `check_index_id` → HTTP 400 |

#### Indexer mode (Chroma collection metadata)

| Function | Prototype | Calls |
|----------|-----------|-------|
| `available_indexers` | `() -> list[str]` | `["chroma"]` if `sparse_backend == "none"`, else `INDEXER_MODES` |
| `read_indexer_mode` | `(index_id) -> str \| None` | `_vector_store(...).try_get_collection()`; normalizes `vector` → `chroma` |
| `write_indexer_mode` | `(index_id, mode) -> None` | `_vector_store(...).modify_metadata` |
| `resolve_ingest_mode` | `(index_id, requested) -> str` | locks indexer after first ingest (409 on mismatch) |

#### Handles / lifecycle

| Function | Prototype | Calls |
|----------|-----------|-------|
| `index_handles` | `(app, index_id, *, cache=True, chroma_embedder=None) -> IndexHandles` | `make_vector_store`, `make_node_store`, `make_sparse_store`, `make_chroma_indexer`, `make_bm25_indexer` |
| `ensure_loaded` | `(h: IndexHandles) -> None` | `h.chroma.load()`, `h.bm25.load()`, `_ensure_chroma_embedder` |
| `startup` | `(app: FastAPI) -> None` | validates `CHUNKERS`; preloads embedders; `make_reranker`; warms `default` index |

#### Search / ingest

| Function | Prototype | Calls |
|----------|-----------|-------|
| `search_index` | `(h, *, mode, query, top_k, rerank, reranker, expand=None) -> tuple[list[dict], list[dict]]` | `chroma.search` / `bm25.search` with `expand`; hybrid → `combine_hybrid_results`; optional `reranker.rerank`; `node_from_retrieved`, `format_retrieved` |
| `ingest_file` | `(app, *, index_id, data, filename, index_description, embedding_model, chunker_name, indexer) -> dict` | `resolve_ingest_mode`, `save_upload`, `make_embedder`, `make_chunker`, `chunk_file`, `_ensure_collection`, `add_chunks` / `remove_source`, `write_indexer_mode` |

#### Index / corpus admin

| Function | Prototype | Calls |
|----------|-----------|-------|
| `list_indices` | `() -> dict` | `list_indices_detailed`, `read_indexer_mode`, sparse `chunk_count` for bm25-only |
| `delete_index` | `(app, index_id) -> None` | `index_handles`, `chroma.delete_index`, `bm25.delete_index` |
| `remove_source` | `(app, index_id, source) -> None` | `chroma.remove_source`, `bm25.remove_source` |

---

## ingest

### `app.ingest.upload`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `save_upload` | `(data: bytes, original_filename: str \| None, corpus_dir: Path, *, max_bytes: int) -> Path` | `orchestration.ingest_file` |

**Calls:** `sanitize_corpus_filename(original_filename)`

### `app.ingest.corpus`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `corpus_dir` | `(corpus_root: Path, index_id: str) -> Path` | `orchestration.ingest_file`, `list_corpus_files`, `unlink_corpus_file` |
| `list_corpus_files` | `(corpus_root: Path, index_id: str) -> list[str]` | `main.list_corpus`, `main.clear_corpus`, `orchestration` (indirect) |
| `unlink_corpus_file` | `(corpus_root: Path, index_id: str, filename: str) -> str` | `main.clear_corpus`, `main.delete_corpus_file` |

### `app.ingest.filename_sanitizer`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `sanitize_corpus_filename` | `(original: str \| None) -> str` | `upload.save_upload`, `main.delete_corpus_file` |

---

## chunkers

### `app.chunkers.chunker_factory`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `CHUNKERS` | constant | `frozenset({"simple", "hierarchical", "markdown", "sentence_window", "semantic"})` |
| `make_chunker` | function | `(name: str, *, chunk_size, chunk_overlap, embed_model=None, hierarchical_parent_multiplier=3, sentence_window_size=3, semantic_breakpoint_percentile=95, semantic_buffer_size=1) -> BaseChunker` |

**Called from:** `orchestration.ingest_file`, `orchestration.startup` (validates against `CHUNKERS`)

**`make_chunker` returns:** `SimpleChunker` \| `MarkdownChunker` \| `HierarchicalChunker` \| `SentenceWindowChunker` \| `SemanticChunker`

### `app.chunkers.base_chunker`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `ChunkSet` | dataclass | `embed_chunks: list[BaseNode]`, `all_chunks: list[BaseNode] \| None` |
| `BaseChunker` | class | `chunk_file(path: Path) -> ChunkSet` |
| | | `chunk_corpus(corpus_dir: Path) -> ChunkSet` |

**`chunk_file` called from:** `orchestration.ingest_file`

### Implementations (subclass `BaseChunker`)

| Module | Class | `name` |
|--------|-------|--------|
| `app.chunkers.simple_chunker` | `SimpleChunker` | `"simple"` |
| `app.chunkers.markdown_chunker` | `MarkdownChunker` (section-based) | `"markdown"` |
| `app.chunkers.hierarchical_chunker` | `HierarchicalChunker` | `"hierarchical"` |
| `app.chunkers.sentence_window_chunker` | `SentenceWindowChunker` | `"sentence_window"` |
| `app.chunkers.semantic_chunker` | `SemanticChunker` | `"semantic"` |

**Created by:** `make_chunker` only

---

## embedders

### `app.embedders.embedder_factory`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `EMBEDDER_BACKENDS` | constant | `frozenset({"huggingface"})` |
| `make_embedder` | function | `(model_name: str, *, backend: str) -> BaseEmbedder` |

**Called from:** `orchestration.startup`, `orchestration.ingest_file`, `orchestration.index_handles` / `_ensure_chroma_embedder`

**Returns:** `HuggingFaceEmbedder` (cached per model name)

### `app.embedders.base_embedder`

| Class | Prototype |
|-------|-----------|
| `BaseEmbedder` | `model_name: str` (property) |
| | `embedding_model: BaseEmbedding` (property) |

### `app.embedders.huggingface_embedder`

| Class | Prototype |
|-------|-----------|
| `HuggingFaceEmbedder` | `__init__(model_name: str)` |

**Used by:** `make_embedder`; passed to `make_chunker` (semantic), `ChromaIndexer.add_chunks`, `ChromaIndexer.bind_embedder`

---

## indexers

### `app.indexers.indexer_factory`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `make_chroma_indexer` | `(index_id, *, embedding_store, lookup_store, embedder=None) -> ChromaIndexer` | `orchestration.index_handles` |
| `make_bm25_indexer` | `(index_id, *, keyword_store, context_store) -> Bm25Indexer` | `orchestration.index_handles` |

### `app.indexers.base_indexer`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `validate_index_id` | function | `(index_id: str) -> None` |
| `IndexerDeps` | dataclass | paths + search flags (`corpus_dir`, `index_store_dir`, defaults, rerank/hybrid flags) |
| `BaseIndexer` | ABC | see methods below |

**`BaseIndexer` methods:**

```
corpus_dir() -> Path
delete_index() -> None
list_corpus_files() -> list[str]
delete_corpus_file(filename: str) -> str
clear_corpus() -> list[str]
load() -> None
bind_embedder(embedder) -> None
bind_reranker(reranker) -> None
add_chunks(chunks: ChunkSet, *, source, embedder, description=None, chunker_name=None) -> int
ready -> bool  (property)
chunk_count() -> int
search(query: str, top_k: int, *, rerank: bool | None = None) -> list[dict]
```

**`validate_index_id` called from:** `orchestration.validate_index_id`, `ChromaIndexer.__init__`, `Bm25Indexer.__init__`, `ingest.corpus.corpus_dir`

### `app.indexers.chroma_indexer`

| Class | Extra |
|-------|-------|
| `ChromaIndexer` | `bind_embedder(embedder: BaseEmbedder)`, `bind_reranker(reranker: BaseReranker)` |

**Uses (injected):** `BaseVectorStore`, `BaseNodeStore`

**Store calls from `ChromaIndexer`:** `vector_store.exists`, `delete_store`, `delete_by_source`, `chunk_count`, `try_get_collection`, `get_collection`, `create_collection`, `resolve_embedding_model`, `resolve_chunker`, `read_description`, `modify_metadata`; `node_store.exists`, `delete_store`, `delete_by_source`, `add_nodes`, `as_llama_docstore`

**Called from `app.main`:** via `IndexHandles.chroma` — `load`, `add_chunks`, `search`, `delete_index`, `list_corpus_files`, `delete_corpus_file`, `bind_embedder`, `chunk_count`, `corpus_dir`, `embedding_model`, `chunker`, `index_metadata`

### `app.indexers.bm25_indexer`

| Class | Extra |
|-------|-------|
| `Bm25Indexer` | `active` (property), `delete_by_source(source: str)` |

**Uses (injected):** `BaseSparseStore`

**Store calls from `Bm25Indexer`:** `sparse_store.active`, `delete_store`, `delete_by_source`, `add_chunks`, `chunk_count`, `load_records`, `load_retriever`

**Called from `app.main`:** via `IndexHandles.bm25` — `load`, `add_chunks`, `search`, `delete_index`, `delete_by_source`, `chunk_count`, `ready`, `active`

---

## stores

### `app.stores.store_factory`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `VECTOR_BACKENDS` | constant | `frozenset({"chroma"})` |
| `NODE_STORE_BACKENDS` | constant | `frozenset({"json", "sqlite"})` |
| `SPARSE_BACKENDS` | constant | `frozenset({"none", "json_bm25", "sqlite_bm25"})` |
| `make_vector_store` | function | `(index_id, *, backend, store_root) -> BaseVectorStore` |
| `make_node_store` | function | `(index_id, *, backend, store_root) -> BaseNodeStore` |
| `make_sparse_store` | function | `(index_id, *, backend, store_root) -> BaseSparseStore` |

**Called from:** `orchestration.index_handles`, `orchestration._vector_store`, `orchestration.list_indices`, `main.set_description`

**Returns:**

| Factory | Implementations |
|---------|-----------------|
| `make_vector_store` | `ChromaVectorStore` |
| `make_node_store` | `JsonNodeStore` \| `SqliteNodeStore` |
| `make_sparse_store` | `NoneSparseStore` \| `JsonBm25SparseStore` \| `SqliteBm25SparseStore` |

### `app.stores.base_vector_store`

| Symbol | Prototype |
|--------|-----------|
| `IndexSnapshotError` | exception |
| `BaseVectorStore` | `exists()`, `delete_store()`, `delete_by_source(source)`, `chunk_count()`, `try_get_collection()`, `get_collection()`, `create_collection(*, embedding_model, chunker, description=None)`, `resolve_embedding_model()`, `resolve_chunker()`, `read_description()`, `modify_metadata(metadata)` |

### `app.stores.chroma_vector_store`

| Symbol | Prototype | Called from |
|--------|-----------|-------------|
| `list_indices_detailed` | `(store_root: Path) -> list[dict]` | `orchestration.list_indices` |
| `write_index_description` | `(index_id, description, store_root) -> dict` | `main.set_description` |
| `ChromaVectorStore` | implements `BaseVectorStore` | `make_vector_store`, `write_index_description` |

### `app.stores.base_node_store`

| Class | Prototype |
|-------|-----------|
| `BaseNodeStore` | `exists()`, `delete_store()`, `delete_by_source(source)`, `add_nodes(nodes)`, `as_llama_docstore()` |

### `app.stores.json_node_store` / `app.stores.sqlite_node_store`

| Class |
|-------|
| `JsonNodeStore` |
| `SqliteNodeStore` |

### `app.stores.base_sparse_store`

| Symbol | Prototype |
|--------|-----------|
| `SparseHit` | dataclass: `chunk_id`, `text`, `score`, `metadata` |
| `BaseSparseStore` | `active` (property), `delete_store()`, `delete_by_source(source)`, `add_chunks(nodes)`, `chunk_count()`, `load_records()`, `load_retriever()` |

### `app.stores.none_sparse_store` / `json_bm25_sparse_store` / `sqlite_bm25_sparse_store`

| Class |
|-------|
| `NoneSparseStore` |
| `JsonBm25SparseStore` |
| `SqliteBm25SparseStore` |

---

## hybrid

### `app.hybrid` (`__init__.py` — single module)

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `combine_hybrid_results` | `(vector_hits, sparse_hits, *, limit, rank_fusion_k=60) -> list[NodeWithScore]` | `orchestration.search_index` |
| `node_from_retrieved` | `(hit: dict) -> NodeWithScore` | `orchestration.search_index` |
| `format_retrieved` | `(hit: NodeWithScore) -> dict` | `orchestration.search_index` |
| `_reciprocal_rank_fusion` | `(rankings, *, k=60) -> list[str]` | `combine_hybrid_results` (internal) |
| `_merge_hybrid_hits` | `(vector_hits, sparse_hits, merged_ids) -> list[NodeWithScore]` | `combine_hybrid_results` (internal) |

---

## rerankers

### `app.rerankers.reranker_factory`

| Symbol | Prototype | Called from |
|--------|-----------|-------------|
| `RERANKER_BACKENDS` | `frozenset({"cross_encoder"})` | — |
| `make_reranker` | `(model_name: str, *, backend: str) -> BaseReranker` | `orchestration.startup` |

**Returns:** `CrossEncoderReranker` (cached per model name)

### `app.rerankers.base_reranker`

| Class | Prototype |
|-------|-----------|
| `BaseReranker` | `model_name` (property), `rerank(hits, query, *, top_n) -> list[NodeWithScore]` |

**`rerank` called from:** `orchestration.search_index` via `app.state.reranker`

### `app.rerankers.cross_encoder_reranker`

| Class |
|-------|
| `CrossEncoderReranker` |

---

## Call summary (main → orchestration → packages)

```
lifespan
  → orchestration.startup
       → settings, CHUNKERS, make_embedder, make_reranker, index_handles, load, _ensure_chroma_embedder

main.ingest / main.retrieve / corpus routes
  → orchestration (ingest_file, search_index, list_indices, delete_index, remove_source, …)

index_handles
  → make_vector_store, make_node_store, make_sparse_store, make_chroma_indexer, make_bm25_indexer

ingest_file
  → validate_index_id, resolve_ingest_mode, save_upload, make_embedder, make_chunker, chunk_file,
     ChromaIndexer.add_chunks / remove_source, Bm25Indexer.add_chunks / remove_source,
     _ensure_collection, write_indexer_mode

search_index
  → read_indexer_mode (via main.retrieve), ChromaIndexer.search | Bm25Indexer.search (expand=…)
       → combine_hybrid_results (hybrid) | BaseReranker.rerank (optional)
       → node_from_retrieved, format_retrieved

delete_index / corpus / files
  → remove_source, unlink_corpus_file, ChromaIndexer.delete_index, Bm25Indexer.delete_index

list_indices
  → list_indices_detailed, read_indexer_mode, make_sparse_store.chunk_count

set_description
  → write_index_description
```

---

## Registries (code limits)

| Registry | Module | Values |
|----------|--------|--------|
| `CHUNKERS` | `chunker_factory` | simple, hierarchical, markdown, sentence_window, semantic |
| `EMBEDDER_BACKENDS` | `embedder_factory` | huggingface |
| `RERANKER_BACKENDS` | `reranker_factory` | cross_encoder |
| `VECTOR_BACKENDS` | `store_factory` | chroma |
| `NODE_STORE_BACKENDS` | `store_factory` | json, sqlite |
| `SPARSE_BACKENDS` | `store_factory` | none, json_bm25, sqlite_bm25 |
| `INDEXER_MODES` | `orchestration` | chroma, bm25, hybrid (`available_indexers` filters when `sparse_backend == "none"`) |

**Env allowlists** (from `settings`): `available_chunkers`, `available_embedding_models`, `*_backend` strings.
