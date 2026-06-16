# Retrieval service — API surface

Structural reference for diagram generation: **packages, modules, classes, function prototypes, and who calls whom**. No implementations.

Root: `retrieval/app/`

---

## config

### `app.config`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `settings` | instance | `Settings` — loaded from `env.toml` `[retrieval]` |
| `Settings` | class | Pydantic settings (paths, backends, allowlists, chunk/rerank/hybrid flags) |

**`settings` fields used elsewhere:** `corpus_dir`, `index_store_dir`, `chunk_size`, `chunk_overlap`, `chunker_name`, `available_chunkers`, `hierarchical_parent_multiplier`, `hierarchical_expand_parent`, `sentence_window_size`, `semantic_breakpoint_percentile`, `semantic_buffer_size`, `available_embedding_models`, `default_embedding_model`, `max_upload_bytes`, `rerank_enabled`, `rerank_model`, `rerank_candidate_multiplier`, `vector_backend`, `node_store_backend`, `embedder_backend`, `reranker_backend`, `sparse_backend`, `hybrid_enabled`, `hybrid_candidate_multiplier`

**Called from:** `app.main` (everywhere settings are read)

---

## main (HTTP entry)

### `app.main`

#### Types / models

| Symbol | Kind | Notes |
|--------|------|-------|
| `RetrieveRequest` | Pydantic model | `query`, `top_k`, `index_id`, `rerank` |
| `RetrievedChunk` | Pydantic model | `chunk_id`, `text`, `score`, `metadata` |
| `RetrieveResponse` | Pydantic model | `query`, `chunks`, `index_id`, `candidate_count`, `candidates` |
| `IngestResponse` | Pydantic model | `index_id`, `saved_as`, `chunks_indexed`, `embedding_model`, `chunker`, `indexer`, `ready` |
| `IndexDescription` | Pydantic model | `description` |
| `IndexHandles` | dataclass | `chroma: ChromaIndexer`, `bm25: Bm25Indexer` |
| `INDEXER_CHOICES` | constant | `("vector", "bm25", "hybrid")` |
| `app` | FastAPI | ASGI app with `lifespan` |

#### Private helpers

| Function | Prototype | Calls |
|----------|-----------|-------|
| `_indexer_deps` | `() -> IndexerDeps` | reads `settings` → builds `IndexerDeps(...)` |
| `_index_handles` | `(app, index_id, *, cache=True) -> IndexHandles` | `make_chroma_indexer(...)`, `make_bm25_indexer(...)`, `make_vector_store(...)`, `make_node_store(...)`, `make_sparse_store(...)` |
| `_delete_index_storage` | `(handles: IndexHandles) -> None` | `handles.chroma.delete_index()`, `handles.bm25.delete_index()` |
| `_available_indexers` | `() -> list[str]` | reads `settings.sparse_backend` |
| `_index_vector_store` | `(index_id, deps) -> BaseVectorStore` | `make_vector_store(index_id, backend=settings.vector_backend, store_root=deps.index_store_dir)` |
| `_read_indexer_choice` | `(index_id, deps) -> str \| None` | `_index_vector_store(...).try_get_collection()` |
| `_write_indexer_choice` | `(index_id, deps, indexer: str) -> None` | `_index_vector_store(...)`, `.get_collection()`, `.modify_metadata(...)` |
| `_search` | `(handles, deps, reranker, query, top_k, *, rerank, mode) -> tuple[list[dict], list[dict]]` | `bm25.search`, `chroma.search`, `node_from_retrieved`, `sparse_hit_from_retrieved`, `combine_hybrid_results`, `reranker.rerank`, `format_retrieved` |
| `_validate_index_id` | `(index_id: str) -> None` | `validate_index_id(index_id)` |
| `_preload_embedding_models` | `() -> None` | `make_embedder(model, backend=settings.embedder_backend)` for each `settings.available_embedding_models` |
| `_bind_embedder` | `(chroma: ChromaIndexer) -> None` | `make_embedder(...)`, `chroma.bind_embedder(...)` |
| `_ensure_loaded` | `(handles: IndexHandles) -> None` | `handles.chroma.load()`, `handles.bm25.load()`, `_bind_embedder(handles.chroma)` |

#### Lifespan

| Function | Prototype | Calls |
|----------|-----------|-------|
| `lifespan` | `(app: FastAPI) -> AsyncIterator[None]` | validates `settings.available_chunkers` ⊆ `CHUNKERS`; `_preload_embedding_models()`; `make_reranker(settings.rerank_model, backend=settings.reranker_backend)`; `_index_handles(app, "default")`; `chroma.load()`, `bm25.load()`, `_bind_embedder(chroma)` |

#### HTTP routes

| Route | Handler | Calls |
|-------|---------|-------|
| `GET /health` | `health()` | — |
| `GET /indices` | `list_indices(request)` | `list_indices_detailed(deps.index_store_dir)`, `_read_indexer_choice`, `make_sparse_store(...).chunk_count()` |
| `GET /ingest/options` | `ingest_options()` | `_available_indexers()`, reads `settings` allowlists/defaults/backends |
| `POST /indices/{index_id}/description` | `set_index_description(...)` | `_validate_index_id`, `write_index_description(...)`, updates `handles.chroma.index_metadata` |
| `DELETE /indices/{index_id}` | `delete_index(...)` | `_validate_index_id`, `_delete_index_storage(handles)` |
| `GET /indices/{index_id}/files` | `list_corpus_files(...)` | `_validate_index_id`, `_index_handles`, `handles.chroma.list_corpus_files()` |
| `DELETE /indices/{index_id}/corpus` | `clear_corpus(...)` | `_validate_index_id`, `_index_handles`, `handles.bm25.delete_by_source(name)`, `handles.chroma.delete_corpus_file(name)` |
| `DELETE /indices/{index_id}/files/{filename}` | `delete_corpus_file(...)` | `_validate_index_id`, `_index_handles`, `sanitize_corpus_filename(filename)`, `handles.bm25.delete_by_source`, `handles.chroma.delete_corpus_file` |
| `POST /retrieve` | `retrieve(...)` | `_validate_index_id`, `_index_handles`, `_ensure_loaded`, `_read_indexer_choice`, `_search(..., request.app.state.reranker, ...)` |
| `POST /ingest` | `ingest(...)` | `_validate_index_id`, `_read_indexer_choice`, `_index_handles`, `save_upload(...)`, `make_embedder(...)`, `make_chunker(...)`, `chunker.chunk_file(saved)`, `handles.chroma.add_chunks(...)` and/or `handles.bm25.add_chunks(...)`, `_index_vector_store(...).create_collection(...)` (bm25-only), `_write_indexer_choice` |

#### CLI

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `run` | `(host, port, *, reload=True) -> None` | `main()` |
| `main` | `(argv=None) -> None` | `__main__` |

---

## ingest

### `app.ingest.upload`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `save_upload` | `(data: bytes, original_filename: str \| None, corpus_dir: Path, *, max_bytes: int) -> Path` | `app.main.ingest` |

**Calls:** `sanitize_corpus_filename(original_filename)`

### `app.ingest.filename_sanitizer`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `sanitize_corpus_filename` | `(original: str \| None) -> str` | `app.ingest.upload.save_upload`, `app.main.delete_corpus_file` |

---

## chunkers

### `app.chunkers.chunker_factory`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `CHUNKERS` | constant | `frozenset({"simple", "hierarchical", "markdown", "sentence_window", "semantic"})` |
| `make_chunker` | function | `(name: str, *, chunk_size, chunk_overlap, embed_model=None, hierarchical_parent_multiplier=3, sentence_window_size=3, semantic_breakpoint_percentile=95, semantic_buffer_size=1) -> BaseChunker` |

**Called from:** `app.main.ingest`, `app.main.lifespan` (validates against `CHUNKERS`)

**`make_chunker` returns:** `SimpleChunker` \| `MarkdownChunker` \| `HierarchicalChunker` \| `SentenceWindowChunker` \| `SemanticChunker`

### `app.chunkers.base_chunker`

| Symbol | Kind | Prototype |
|--------|------|-----------|
| `ChunkSet` | dataclass | `embed_chunks: list[BaseNode]`, `all_chunks: list[BaseNode] \| None` |
| `BaseChunker` | class | `chunk_file(path: Path) -> ChunkSet` |
| | | `chunk_corpus(corpus_dir: Path) -> ChunkSet` |

**`chunk_file` called from:** `app.main.ingest`

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

**Called from:** `app.main._preload_embedding_models`, `app.main._bind_embedder`, `app.main.ingest`

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
| `make_chroma_indexer` | `(index_id, deps, *, vector_store, node_store) -> ChromaIndexer` | `app.main._index_handles` |
| `make_bm25_indexer` | `(index_id, deps, *, sparse_store) -> Bm25Indexer` | `app.main._index_handles` |

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

**`validate_index_id` called from:** `app.main._validate_index_id`, `ChromaIndexer.__init__`, `Bm25Indexer.__init__`

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

**Called from:** `app.main._index_handles`, `app.main._index_vector_store`, `app.main.list_indices`

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
| `list_indices_detailed` | `(store_root: Path) -> list[dict]` | `app.main.list_indices` |
| `write_index_description` | `(index_id, description, store_root) -> dict` | `app.main.set_index_description` |
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

### `app.hybrid.combine`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `combine_hybrid_results` | `(vector_hits: list[NodeWithScore], sparse_hits: list, *, limit: int, rank_fusion_k=60) -> list[NodeWithScore]` | `app.main._search` |

**Calls:** `reciprocal_rank_fusion(...)`, `merge_hybrid_hits(...)`

### `app.hybrid.merge_hits`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `node_from_retrieved` | `(hit: dict) -> NodeWithScore` | `app.main._search` |
| `sparse_hit_from_retrieved` | `(hit: dict) -> SparseHit` | `app.main._search` |
| `format_retrieved` | `(hit: NodeWithScore) -> dict` | `app.main._search` |
| `merge_hybrid_hits` | `(vector_hits, sparse_hits, merged_ids) -> list[NodeWithScore]` | `combine_hybrid_results` |

### `app.hybrid.rank_fusion`

| Function | Prototype | Called from |
|----------|-----------|-------------|
| `reciprocal_rank_fusion` | `(rankings: list[list[str]], *, k=60) -> list[str]` | `combine_hybrid_results` |

---

## rerankers

### `app.rerankers.reranker_factory`

| Symbol | Prototype | Called from |
|--------|-----------|-------------|
| `RERANKER_BACKENDS` | `frozenset({"cross_encoder"})` | — |
| `make_reranker` | `(model_name: str, *, backend: str) -> BaseReranker` | `app.main.lifespan` |

**Returns:** `CrossEncoderReranker` (cached per model name)

### `app.rerankers.base_reranker`

| Class | Prototype |
|-------|-----------|
| `BaseReranker` | `model_name` (property), `rerank(hits, query, *, top_n) -> list[NodeWithScore]` |

**`rerank` called from:** `app.main._search` via `request.app.state.reranker`

### `app.rerankers.cross_encoder_reranker`

| Class |
|-------|
| `CrossEncoderReranker` |

---

## Call summary (main → packages)

```
lifespan
  → settings, CHUNKERS, make_embedder, make_reranker, _index_handles, ChromaIndexer.load, Bm25Indexer.load, bind_embedder

_index_handles
  → make_vector_store, make_node_store, make_sparse_store, make_chroma_indexer, make_bm25_indexer

ingest
  → validate_index_id, save_upload, make_embedder, make_chunker, BaseChunker.chunk_file,
     ChromaIndexer.add_chunks, Bm25Indexer.add_chunks, ChromaVectorStore.create_collection (bm25-only)

retrieve
  → _index_handles, _ensure_loaded, _read_indexer_choice, _search
       → ChromaIndexer.search | Bm25Indexer.search | combine_hybrid_results | BaseReranker.rerank
       → node_from_retrieved, sparse_hit_from_retrieved, format_retrieved

delete_index / corpus / files
  → ChromaIndexer.delete_index | delete_corpus_file | list_corpus_files
  → Bm25Indexer.delete_index | delete_by_source

list_indices
  → list_indices_detailed, _read_indexer_choice, make_sparse_store.chunk_count

set_index_description
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
| `INDEXER_CHOICES` | `main` | vector, bm25, hybrid (filtered by `sparse_backend`) |

**Env allowlists** (from `settings`): `available_chunkers`, `available_embedding_models`, `*_backend` strings.
