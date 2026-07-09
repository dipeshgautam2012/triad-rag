# Retrieval workflow — Gemini diagram (main + orchestration + packages)

> **Diagram source of truth** for Gemini image generation. HTTP in `main.py`; ingest/search in `orchestration.py`. See [`retrieval-workflow-surface.md`](retrieval-workflow-surface.md) and [`DESIGN.md`](../../DESIGN.md).

Generate from the **user layout sketch** + this spec. Attach the sketch image when prompting Gemini.

**Sketch:** [`retrieval-workflow-sketch.png`](../../dump/retrieval-workflow-sketch.png) — internal layout: factory | base+implementations inside each package container.

**API truth:** [`retrieval-workflow-surface.md`](retrieval-workflow-surface.md)

**Authoring note:** Words like “Col 1”, “Container 1”, “Row 2” in this spec are for **you only** — they must **never** appear on the generated image.

## 1. Mental model

### Overall layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADER: env.toml ──loads──→ settings    code registries (no arrow)       │
├──────────────┬──────────────────────────────────────────────────────────┤
│  main.py     │  ┌─ indexers/ ─────────────────────────────────────────┐ │
│  HTTP routes │  │ indexer_factory │ BaseIndexer ┊ Chroma · Bm25      │ │
│      │       │  └────────────────────────────────────────────────────┘ │
│ delegates    │  ┌─ embedders/ ────────────────────────────────────────┐ │
│      ↓       │  │ embedder_factory │ BaseEmbedder ┊ HuggingFace      │ │
│ orchestration│  └────────────────────────────────────────────────────┘ │
│  ingest ·    │  ┌─ chunkers/ ─ ... ────────────────────────────────────┐ │
│  search ·    │  ┌─ stores/ ─ ... ─────────────────────────────────────┐ │
│  handles     │  ┌─ rerankers/ ─ ... ──────────────────────────────────┐ │
│              │  ┌─ ingest/ ─ ... ──────────────────────────────────────┐ │
│  ──→ each    │  ┌─ hybrid/ ─ ... ─────────────────────────────────────┐ │
│  container   │                                                          │
│  reads       │                                                          │
│  settings ──→│                                                          │
└──────────────┴──────────────────────────────────────────────────────────┘
```

### Package container (repeat for each module)

Each package is one **bordered container** with:

1. **Header bar** on top: package path (e.g. `indexers/`, `stores/`)
2. **Inside** the border — same sketch layout, two zones:
   - **Left:** factory box (`indexer_factory`, `make_*` functions)
   - **Right:** base box on top → **dashed vertical** ↓ → implementation box(es) below

Factory + base + derived all live **inside** the same container. The container border wraps all three.

**Stores container:** inside border, still `store_factory` left; right side has three sub-columns (vector | node | sparse), each with base ┊ impls.

**ingest / hybrid containers:** header + flat function list inside (no factory, no base/impl split).

### Rules

1. **Left:** single tall `main.py` box.
2. **Right:** stack of **7 package containers** (one per module).
3. **Inside each factory container:** factory (left) | base + derived (right); dashed base→impl only.
4. **No** `creates` arrows inside containers. **No** factory→base line.
5. **main.py → each package container** — one solid arrow per container (9 arrows from main total: settings + 7 packages). See §3.
6. **Title:** `Retrieval — main.py workflow` only.
7. **No** outer “packages” mega-wrapper around all containers — each module has its **own** border only.
8. **No layout chrome on the image** — never write `COLUMN 1`, `COLUMN 2`, `ROW 1`, `Container 1`, `#1`, grid coordinates, or spec section numbers. Use real names only (`main.py`, `indexers/`, `indexer_factory`, etc.).

### Colors (match sketch spirit, slightly refined)

| Role | Fill |
|------|------|
| main.py | light blue `#E3F2FD` |
| factories | light indigo `#E8EAF6` |
| base classes | light teal `#E0F2F1` |
| implementations | white `#FFFFFF` with gray border |
| config header | light orange `#FFF3E0` |
| container shell | light gray `#ECEFF1` border `#546E7A` |
| ingest/hybrid interior | light purple `#F3E5F5` |

---

## 2. Module inventory (all rows — nothing omitted)

| # | Container header | Inside border (left \| right) |
|---|------------------|-------------------------------|
| H | **config** (header band, not a container) | `env.toml` · `settings` · `code registries` |
| 1 | **indexers/** | `indexer_factory` \| `BaseIndexer` ┊ `ChromaIndexer` · `Bm25Indexer` |
| 2 | **embedders/** | `embedder_factory` \| `BaseEmbedder` ┊ `HuggingFaceEmbedder` |
| 3 | **chunkers/** | `chunker_factory` \| `BaseChunker` ┊ 5 chunker classes |
| 4 | **stores/** | `store_factory` \| vector \| node \| sparse mini-stacks |
| 5 | **rerankers/** | `reranker_factory` \| `BaseReranker` ┊ `CrossEncoderReranker` |
| 6 | **ingest/** | flat: `save_upload` · `sanitize_corpus_filename` · `corpus_dir` · `list_corpus_files` · `unlink_corpus_file` |
| 7 | **hybrid/** | single `__init__.py`: `combine_hybrid_results` · `node_from_retrieved` · `format_retrieved` · RRF helpers |

**Left column:** `main.py` (HTTP routes) stacked above `orchestration.py` (ingest/search/handles). See §2.1.

---

## 2.1 Left column — `main.py` + `orchestration.py`

### `main.py (HTTP routes)`

| Method | Route | Handler | Delegates to |
|--------|-------|---------|--------------|
| GET | `/health` | `health()` | — |
| GET | `/indices` | `get_indices` | `list_indices` |
| GET | `/ingest/options` | `ingest_options` | `available_indexers`, `settings` |
| POST | `/ingest` | `ingest` | `ingest_file` |
| POST | `/retrieve` | `retrieve` | `index_handles`, `ensure_loaded`, `search_index` |
| POST | `/indices/{index_id}/description` | `set_description` | `write_index_description` |
| DELETE | `/indices/{index_id}` | `delete_index_route` | `delete_index` |
| GET | `/indices/{index_id}/files` | `list_corpus` | `list_corpus_files` |
| DELETE | `/indices/{index_id}/corpus` | `clear_corpus` | `remove_source`, `unlink_corpus_file` |
| DELETE | `/indices/{index_id}/files/{filename}` | `delete_corpus_file` | `remove_source`, `unlink_corpus_file` |

**Models (muted line):** `RetrieveRequest` (`query`, `top_k`, `index_id`, `rerank`, `expand`) · `RetrieveResponse` · `IngestResponse`

### `orchestration.py (wire)`

**Helpers** (one muted line in box):

`index_handles` · `search_index` · `ingest_file` · `ensure_loaded` · `startup` · `read_indexer_mode` · `write_indexer_mode` · `resolve_ingest_mode` · `available_indexers` · `list_indices` · `delete_index` · `remove_source`

**Startup (`startup`, called from `lifespan`):** validates `CHUNKERS`, preloads embedders, `make_reranker`, warms `default` index

**Indexer ids:** `chroma` · `bm25` · `hybrid` (`vector` normalized to `chroma`)

---

## 2.2 Config header (required — not optional)

Thin band **above** the main + containers layout:

| Box | Text |
|-----|------|
| `env.toml` | `[retrieval]` section |
| `settings` | `config.py` — paths · `*_backend` · `available_chunkers` · `available_embedding_models` · defaults · rerank/hybrid flags |
| `code registries` | `CHUNKERS` · `EMBEDDER_BACKENDS` · `RERANKER_BACKENDS` · `VECTOR_BACKENDS` · `NODE_STORE_BACKENDS` · `SPARSE_BACKENDS` · `INDEXER_MODES` |

---

## 2.3 Container content (inside each border)

Each subsection = one bordered container. **Left** = factory (if any). **Right** = base ┊ derived.

### Container 1: `indexers/`

**Left — indexer_factory**
- `make_chroma_indexer`
- `make_bm25_indexer`

**Right — base + derived**
- **BaseIndexer** (+ `validate_index_id`)
- dashed ↓
- Implementation box:
  - **ChromaIndexer** — `load` · `bind_embedder` · `add_chunks` · `search` · `delete_index` · `remove_source`
  - **Bm25Indexer** — `load` · `add_chunks` · `search` · `delete_index` · `remove_source` · `delete_by_source`
  - *index metadata via Chroma collection (embedding_model, chunker, description, indexer)*

### Container 2: `embedders/`

**Left — embedder_factory:** `make_embedder`

**Right:** **BaseEmbedder** (`embedding_model`) ┊ dashed ↓ **HuggingFaceEmbedder**

### Container 3: `chunkers/`

**Left — chunker_factory:** `make_chunker` · `CHUNKERS`

**Right:** **BaseChunker** (`chunk_file` → ChunkSet) ┊ dashed ↓ bullets:
- SimpleChunker · MarkdownChunker (section-based) · HierarchicalChunker · SentenceWindowChunker · SemanticChunker

### Container 4: `stores/`

**Left — store_factory**
- `make_vector_store` · `make_node_store` · `make_sparse_store`

**Right — three sub-columns**

| vector | node | sparse |
|--------|------|--------|
| **BaseVectorStore** `delete_store` · `delete_by_source` | **BaseNodeStore** same | **BaseSparseStore** same |
| ┊ **ChromaVectorStore** *collection metadata* | ┊ **JsonNodeStore** · **SqliteNodeStore** | ┊ **NoneSparseStore** · **JsonBm25SparseStore** · **SqliteBm25SparseStore** |

Note under vector: `list_indices_detailed` · `write_index_description`

### Container 5: `rerankers/`

**Left — reranker_factory:** `make_reranker`

**Right:** **BaseReranker** (`rerank`) ┊ dashed ↓ **CrossEncoderReranker**

### Container 6: `ingest/` (flat inside border)

`save_upload` · `sanitize_corpus_filename` · `corpus_dir` · `list_corpus_files` · `unlink_corpus_file`

### Container 7: `hybrid/` (flat inside border — single module)

`combine_hybrid_results` · `node_from_retrieved` · `format_retrieved` · `_reciprocal_rank_fusion` · `_merge_hybrid_hits`

---

## 2.4 Metadata (box text only — no extra arrows)

Metadata is **not** a separate row. Add as **small notes inside existing boxes**:

### Index metadata (Chroma collection — `ChromaVectorStore`)

Stored on the vector collection at ingest; read by `GET /indices` and `POST /retrieve`:

| Key | Set by | Used for |
|-----|--------|----------|
| `embedding_model` | `create_collection` / ingest | lock model on re-ingest |
| `chunker` | `create_collection` / ingest | lock chunker on re-ingest |
| `description` | ingest or `write_index_description` | index label in `/indices` |
| `indexer` | `write_indexer_mode` after ingest | search mode: chroma / bm25 / hybrid |

**Functions (already in diagram):** `create_collection` · `modify_metadata` · `resolve_embedding_model` · `resolve_chunker` · `read_description` · `read_indexer_mode` · `write_indexer_mode` · `write_index_description`

**Where to show:** small note under **ChromaVectorStore** or **ChromaIndexer**: *collection metadata: embedding_model, chunker, description, indexer*

### Chunk metadata (per passage — set at ingest, returned on search)

Attached to each chunk node; flows through search → hybrid → rerank → `POST /retrieve`:

| Field | When |
|-------|------|
| `source` | corpus filename |
| `file_type` | txt / pdf |
| `page` | PDF pages |
| `chunk_role` · `parent_id` | hierarchical chunking |

**Where to show:** one muted line in **`main.py`** box: *RetrieveResponse: chunk_id, text, score, metadata* — or note on **BaseChunker**: *chunk metadata at ingest*

### No metadata arrows

Do not draw arrows for metadata flow. Arrow count stays per §3 only.

---

## 3. Arrows — simple, one per container

**Arrows hit container borders**, not inner factory/base/impl boxes.

### Config (1 arrow)

| From | To | Label |
|------|-----|-------|
| `env.toml` | `settings` | `loads` |

### From left column (9 arrows — plus config above)

| From | To | Label |
|------|-----|-------|
| `orchestration.py` | `settings` (header) | `reads settings` |
| `main.py` | `orchestration.py` | `delegates` |
| `orchestration.py` | **`indexers/` container** | `indexers` |
| `orchestration.py` | **`embedders/` container** | `embedders` |
| `orchestration.py` | **`chunkers/` container** | `chunkers` |
| `orchestration.py` | **`stores/` container** | `stores` |
| `orchestration.py` | **`rerankers/` container** | `rerankers` |
| `orchestration.py` | **`ingest/` container** | `ingest` |
| `orchestration.py` | **`hybrid/` container** | `hybrid` |

**Total labeled arrows: 10** (1 config `loads` + 9 above). Use short labels — no function lists on arrows.

### Structural lines only (not arrows)

- Dashed vertical **base → derived** inside each factory container
- No `creates` arrows inside containers
- No arrows from `settings` down to containers
- No arrows inside ingest/hybrid containers
- No arrow to `code registries` box

### Do NOT draw or write on the image

- Arrows into `ChromaIndexer` / inner classes (stop at container border)
- Extra arrows beyond the 10 listed
- `A1` / `A2` / `triad-rag` / `Stack`
- **Layout labels:** `COLUMN 1`, `COLUMN 2`, `ROW 1`, `ROW 2`, `Container 1`, `#1`, `LEFT`, `RIGHT`, `grid`, or any spec numbering from this document

**Legend (tiny):** `—— calls` · `- - - extends`

---

## 4. Gemini master prompt (copy + attach sketch)

```
You are redrawing an architecture diagram. I am attaching my LAYOUT SKETCH — copy its internal pattern (factory left | base+derived right).

TASK: Expand to the full retrieval service using PACKAGE CONTAINERS.

LAYOUT:
- LEFT: stacked boxes — "main.py (HTTP routes)" on top, "orchestration.py (wire)" below
- RIGHT: stack of 7 BORDERED CONTAINERS (one per package), each with a header bar
- TOP: config header band — env.toml, settings, code registries (not inside a container)

PACKAGE CONTAINER PATTERN (indexers, embedders, chunkers, stores, rerankers):
- Draw a rounded rectangle BORDER around the whole package
- Header on border top: e.g. "indexers/"
- INSIDE the border only:
  - LEFT: factory box (indexer_factory, make_* functions)
  - RIGHT: Base class on top, dashed vertical line down, implementation box(es) below
- Factory + Base + derived are ALL inside the same container — not floating outside

STORES container: same border; inside: store_factory left; right = 3 sub-columns (vector | node | sparse)

INGEST / HYBRID containers: border + header; flat function list inside (no factory/base split)

TITLE: "Retrieval — API workflow"

CONTAINERS (top to bottom):
1. indexers/ — indexer_factory | BaseIndexer ┊ ChromaIndexer, Bm25Indexer (+ metadata note)
2. embedders/ — embedder_factory | BaseEmbedder ┊ HuggingFaceEmbedder
3. chunkers/ — chunker_factory | BaseChunker ┊ 5 chunker classes
4. stores/ — store_factory | vector/node/sparse mini-stacks
5. rerankers/ — reranker_factory | BaseReranker ┊ CrossEncoderReranker
6. ingest/ — save_upload, sanitize_corpus_filename, corpus_dir, list_corpus_files, unlink_corpus_file
7. hybrid/ — combine_hybrid_results, node_from_retrieved, format_retrieved, RRF helpers (single __init__.py)

main.py routes (all 10): GET /health, GET /indices, GET /ingest/options, POST /ingest, POST /retrieve, POST /indices/{id}/description, DELETE /indices/{id}, DELETE .../corpus, DELETE .../files/{filename}

orchestration.py: index_handles, ingest_file, search_index (expand/rerank), startup, read/write_indexer_mode

ARROWS — exactly 10, hit CONTAINER BORDERS or settings/main/orchestration boxes:
1. env.toml → settings: "loads"
2. orchestration.py → settings: "reads settings"
3. main.py → orchestration.py: "delegates"
4. orchestration.py → indexers/ container: "indexers"
5. orchestration.py → embedders/ container: "embedders"
6. orchestration.py → chunkers/ container: "chunkers"
7. orchestration.py → stores/ container: "stores"
8. orchestration.py → rerankers/ container: "rerankers"
9. orchestration.py → ingest/ container: "ingest"
10. orchestration.py → hybrid/ container: "hybrid"

Inside containers: dashed vertical base→derived only. NO creates arrows. NO arrows to inner classes.

FORBIDDEN: one mega "packages" wrapper; arrows to ChromaIndexer; Stack; A1; triad-rag; extra arrows beyond 10; layout labels (COLUMN 1, ROW 1, Container 1, #1, LEFT/RIGHT grid text — use real module names only).

Output one diagram image.
```

---

## 5. Repair prompt (if Gemini drifts from sketch)

```
Redraw with PACKAGE CONTAINERS.

Fix:
1. Each module (indexers, embedders, chunkers, stores, rerankers, ingest, hybrid) has its own BORDERED container with header
2. Inside each factory container: factory (left) + base + derived (right) — all inside the border
3. LEFT column: main.py (routes) above orchestration.py (wire); main → orchestration: delegates
4. orchestration.py → one arrow to EACH container border (labels: indexers, embedders, chunkers, stores, rerankers, ingest, hybrid)
5. orchestration.py → settings: reads settings; env.toml → settings: loads
6. Dashed base→derived inside containers only; no creates arrows; no arrows to inner classes
7. No single mega "packages" wrapper around everything
8. Remove COLUMN 1, ROW 1, Container 1, #1, or any grid/layout labels — use real names only (main.py, orchestration.py, indexers/, etc.)

Keep title: Retrieval — API workflow
```

---

## 6. Checklist

**Containers**
- [ ] 7 bordered containers with headers: indexers · embedders · chunkers · stores · rerankers · ingest · hybrid
- [ ] Inside each factory container: factory + base + derived **inside** the border
- [ ] Store container: 3 sub-columns inside border

**main.py + orchestration.py**
- [ ] All 10 HTTP routes in main.py
- [ ] orchestration helpers + indexer ids (chroma/bm25/hybrid)
- [ ] Chunk metadata note

**Arrows (10 total)**
- [ ] `loads` · `reads settings` · `delegates`
- [ ] orchestration → each of 7 containers (short labels)
- [ ] Arrows touch **container border**, not inner boxes
- [ ] Dashed base→derived inside only; no creates arrows

**Layout**
- [ ] main.py + orchestration.py stacked on left; no mega packages wrapper
- [ ] No Stack / A1 / triad-rag
- [ ] No COLUMN/ROW/Container numbering on the image
