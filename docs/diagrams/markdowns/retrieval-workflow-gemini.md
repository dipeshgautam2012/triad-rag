# Retrieval workflow — Gemini diagram (main + package containers)

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
│              │  ┌─ indexers/ ─────────────────────────────────────────┐ │
│              │  │ indexer_factory │ BaseIndexer ┊ Chroma · Bm25      │ │
│  main.py     │  └────────────────────────────────────────────────────┘ │
│  API entry   │  ┌─ embedders/ ────────────────────────────────────────┐ │
│              │  │ embedder_factory │ BaseEmbedder ┊ HuggingFace      │ │
│  ──→ each    │  └────────────────────────────────────────────────────┘ │
│  container   │  ┌─ chunkers/ ─ ... ────────────────────────────────────┐ │
│              │  ┌─ stores/ ─ ... ─────────────────────────────────────┐ │
│              │  ┌─ rerankers/ ─ ... ──────────────────────────────────┐ │
│              │  ┌─ ingest/ ─ ... ──────────────────────────────────────┐ │
│              │  ┌─ hybrid/ ─ ... ─────────────────────────────────────┐ │
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
| 6 | **ingest/** | flat: `save_upload` · `sanitize_corpus_filename` |
| 7 | **hybrid/** | flat: `combine_hybrid_results` · merge helpers · `reciprocal_rank_fusion` |

**Left column:** `main.py` — all HTTP routes + helpers (see §2.1).

---

## 2.1 `main.py (API entry)` — full routes

| Method | Route | Handler | Key calls (for arrow labels) |
|--------|-------|---------|------------------------------|
| GET | `/health` | `health()` | — |
| GET | `/indices` | `list_indices` | `list_indices_detailed`, `_read_indexer_choice`, `make_sparse_store` |
| GET | `/ingest/options` | `ingest_options` | `_available_indexers`, `settings` |
| POST | `/ingest` | `ingest` | `save_upload`, `make_embedder`, `make_chunker`, `chunk_file`, `add_chunks` |
| POST | `/retrieve` | `retrieve` | `_index_handles`, `_ensure_loaded`, `_search`, `reranker.rerank` |
| POST | `/indices/{index_id}/description` | `set_index_description` | `write_index_description` |
| DELETE | `/indices/{index_id}` | `delete_index` | `_delete_index_storage` |
| GET | `/indices/{index_id}/files` | `list_corpus_files` | `chroma.list_corpus_files` |
| DELETE | `/indices/{index_id}/corpus` | `clear_corpus` | `bm25.delete_by_source`, `chroma.delete_corpus_file` |
| DELETE | `/indices/{index_id}/files/{filename}` | `delete_corpus_file` | `sanitize_corpus_filename`, `bm25.delete_by_source`, `chroma.delete_corpus_file` |

**Helpers** (one muted line in box, not separate boxes):

`_index_handles` · `_search` · `_bind_embedder` · `_ensure_loaded` · `_read_indexer_choice` · `_write_indexer_choice` · `_validate_index_id` · `_preload_embedding_models`

**Startup (`lifespan`):** validates `CHUNKERS`, `_preload_embedding_models`, `make_reranker`, `_index_handles`, `load`, `_bind_embedder`

---

## 2.2 Config header (required — not optional)

Thin band **above** the main + containers layout:

| Box | Text |
|-----|------|
| `env.toml` | `[retrieval]` section |
| `settings` | `config.py` — paths · `*_backend` · `available_chunkers` · `available_embedding_models` · defaults · rerank/hybrid flags |
| `code registries` | `CHUNKERS` · `EMBEDDER_BACKENDS` · `RERANKER_BACKENDS` · `VECTOR_BACKENDS` · `NODE_STORE_BACKENDS` · `SPARSE_BACKENDS` · `INDEXER_CHOICES` |

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
  - **ChromaIndexer** — `load` · `bind_embedder` · `add_chunks` · `search` · `delete_index` · `delete_corpus_file`
  - **Bm25Indexer** — `load` · `add_chunks` · `search` · `delete_index` · `delete_by_source`
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

`save_upload` · `sanitize_corpus_filename`

### Container 7: `hybrid/` (flat inside border)

`combine_hybrid_results` · `merge_hybrid_hits` · `reciprocal_rank_fusion` · `node_from_retrieved` · `sparse_hit_from_retrieved` · `format_retrieved`

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
| `indexer` | `_write_indexer_choice` after ingest | search mode: vector / bm25 / hybrid |

**Functions (already in diagram):** `create_collection` · `modify_metadata` · `resolve_embedding_model` · `resolve_chunker` · `read_description` · `_read_indexer_choice` · `_write_indexer_choice` · `write_index_description`

**Where to show:** small note under **ChromaVectorStore** or **ChromaIndexer**: *collection metadata: embedding_model, chunker, description, indexer*

### Chunk metadata (per passage — set at ingest, returned on search)

Attached to each chunk node; flows through search → hybrid → rerank → `POST /retrieve`:

| Field | When |
|-------|------|
| `source` | corpus filename |
| `file_type` | txt / pdf |
| `page` | PDF pages |
| `chunk_role` · `parent_id` | hierarchical chunking |

**Where to show:** one muted line in **main.py** box: *RetrieveResponse: chunk_id, text, score, metadata* — or note on **BaseChunker**: *chunk metadata at ingest*

### No metadata arrows

Do not draw arrows for metadata flow. Arrow count stays per §3 only.

---

## 3. Arrows — simple, one per container

**Arrows hit container borders**, not inner factory/base/impl boxes.

### Config (1 arrow)

| From | To | Label |
|------|-----|-------|
| `env.toml` | `settings` | `loads` |

### From main.py (8 arrows — one per target)

| From | To | Label |
|------|-----|-------|
| `main.py` | `settings` (header) | `reads settings` |
| `main.py` | **`indexers/` container** | `indexers` |
| `main.py` | **`embedders/` container** | `embedders` |
| `main.py` | **`chunkers/` container** | `chunkers` |
| `main.py` | **`stores/` container** | `stores` |
| `main.py` | **`rerankers/` container** | `rerankers` |
| `main.py` | **`ingest/` container** | `ingest` |
| `main.py` | **`hybrid/` container** | `hybrid` |

**Total labeled arrows: 9** (1 config + 8 from main). Use short labels above — no function lists on arrows.

### Structural lines only (not arrows)

- Dashed vertical **base → derived** inside each factory container
- No `creates` arrows inside containers
- No arrows from `settings` down to containers
- No arrows inside ingest/hybrid containers
- No arrow to `code registries` box

### Do NOT draw or write on the image

- Arrows into `ChromaIndexer` / inner classes (stop at container border)
- Extra arrows beyond the 9 listed
- `A1` / `A2` / `triad-rag` / `Stack`
- **Layout labels:** `COLUMN 1`, `COLUMN 2`, `ROW 1`, `ROW 2`, `Container 1`, `#1`, `LEFT`, `RIGHT`, `grid`, or any spec numbering from this document

**Legend (tiny):** `—— calls` · `- - - extends`

---

## 4. Gemini master prompt (copy + attach sketch)

```
You are redrawing an architecture diagram. I am attaching my LAYOUT SKETCH — copy its internal pattern (factory left | base+derived right).

TASK: Expand to the full retrieval service using PACKAGE CONTAINERS.

LAYOUT:
- LEFT: tall "main.py (API entry)" box with all HTTP routes + helpers
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

TITLE: "Retrieval — main.py workflow"

CONTAINERS (top to bottom):
1. indexers/ — indexer_factory | BaseIndexer ┊ ChromaIndexer, Bm25Indexer (+ metadata note)
2. embedders/ — embedder_factory | BaseEmbedder ┊ HuggingFaceEmbedder
3. chunkers/ — chunker_factory | BaseChunker ┊ 5 chunker classes
4. stores/ — store_factory | vector/node/sparse mini-stacks
5. rerankers/ — reranker_factory | BaseReranker ┊ CrossEncoderReranker
6. ingest/ — save_upload, sanitize_corpus_filename
7. hybrid/ — combine_hybrid_results, merge_hybrid_hits, reciprocal_rank_fusion, helpers

main.py routes (all 10): GET /health, GET /indices, GET /ingest/options, POST /ingest, POST /retrieve, POST /indices/{id}/description, DELETE /indices/{id}, DELETE .../corpus, DELETE .../files/{filename}

ARROWS — exactly 9, hit CONTAINER BORDERS or settings box:
1. env.toml → settings: "loads"
2. main.py → settings: "reads settings"
3. main.py → indexers/ container: "indexers"
4. main.py → embedders/ container: "embedders"
5. main.py → chunkers/ container: "chunkers"
6. main.py → stores/ container: "stores"
7. main.py → rerankers/ container: "rerankers"
8. main.py → ingest/ container: "ingest"
9. main.py → hybrid/ container: "hybrid"

Inside containers: dashed vertical base→derived only. NO creates arrows. NO arrows to inner classes.

FORBIDDEN: one mega "packages" wrapper; arrows to ChromaIndexer; Stack; A1; triad-rag; extra arrows beyond 9; layout labels (COLUMN 1, ROW 1, Container 1, #1, LEFT/RIGHT grid text — use real module names only).

Output one diagram image.
```

---

## 5. Repair prompt (if Gemini drifts from sketch)

```
Redraw with PACKAGE CONTAINERS.

Fix:
1. Each module (indexers, embedders, chunkers, stores, rerankers, ingest, hybrid) has its own BORDERED container with header
2. Inside each factory container: factory (left) + base + derived (right) — all inside the border
3. main.py → one arrow to EACH container border (labels: indexers, embedders, chunkers, stores, rerankers, ingest, hybrid)
4. main.py → settings: reads settings; env.toml → settings: loads
5. Dashed base→derived inside containers only; no creates arrows; no arrows to inner classes
6. No single mega "packages" wrapper around everything
7. Remove COLUMN 1, ROW 1, Container 1, #1, or any grid/layout labels — use real names only (main.py, indexers/, etc.)

Keep title: Retrieval — main.py workflow
```

---

## 6. Checklist

**Containers**
- [ ] 7 bordered containers with headers: indexers · embedders · chunkers · stores · rerankers · ingest · hybrid
- [ ] Inside each factory container: factory + base + derived **inside** the border
- [ ] Store container: 3 sub-columns inside border

**main.py**
- [ ] All 10 HTTP routes listed
- [ ] Helpers + chunk metadata note

**Arrows (9 total)**
- [ ] `loads` · `reads settings`
- [ ] main → each of 7 containers (short labels)
- [ ] Arrows touch **container border**, not inner boxes
- [ ] Dashed base→derived inside only; no creates arrows

**Layout**
- [ ] main.py tall on left; no mega packages wrapper
- [ ] No Stack / A1 / triad-rag
- [ ] No COLUMN/ROW/Container numbering on the image
