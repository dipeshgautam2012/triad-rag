# Retrieval metadata — Gemini diagram (structure only)

Generate a **standalone schema diagram** for retrieval metadata. This is **not** the main workflow diagram — no `main.py`, factories, or package containers.

**API truth:** [`DESIGN.md` § Retrieval](../../../docs/DESIGN.md) · [`retrieval-workflow-surface.md`](retrieval-workflow-surface.md)

**Authoring note:** Section numbers and “Tier 1” labels in this doc are for **you only** — never on the generated image.

---

## 1. Mental model

Three **stacked tiers** (top → bottom). Each tier is one bordered box with a header bar.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Retrieval — metadata structure                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ Service config (env.toml · startup) ─────────────────────────────┐  │
│  │ sparse_backend · node_store_backend · vector_backend              │  │
│  │ rerank_enabled · rerank_model · rerank_candidate_multiplier     │  │
│  │ hybrid_candidate_multiplier · sentence_window_size · …          │  │
│  │ muted: service-wide — not stored per index                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Index metadata (Chroma collection · one per index_id) ──────────┐  │
│  │ embedding_model · chunker · indexer · description               │  │
│  │ set on first ingest · indexer locked on re-ingest               │  │
│  │ read by GET /indices · POST /retrieve (search mode)             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Chunk metadata (per passage · set at chunking) ────────────────┐  │
│  │ always: source · file_type                                        │  │
│  │ pdf: page                                                         │  │
│  │ hierarchical: chunk_role · level · parent_id                      │  │
│  │ sentence_window: window · original_text                           │  │
│  │ returned on POST /retrieve as RetrievedChunk.metadata             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Where chunk metadata is copied (muted footer band) ────────────┐  │
│  │ Chroma embedding · node store (parents) · sparse BM25 record    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Diagram type

- **Entity / schema layout** — field lists inside boxes, not a flowchart of modules.
- **No arrows between tiers** (optional: one faint down-arrow between tiers is OK; do not draw ingest→store→retrieve flows).
- **No** `main.py`, factories, `COLUMN 1`, `ROW 1`, `triad-rag`, `Stack`, `A1`, or layout grid labels.

---

## 2. Tier content (exact fields)

### Tier A — Service config (`env.toml`, loaded at startup)

Muted subtitle: *service-wide — not per index*

| Field | Role |
|-------|------|
| `sparse_backend` | `none` / `json_bm25` / `sqlite_bm25` — which BM25 backends exist |
| `node_store_backend` | `json` / `sqlite` — parent nodes for hierarchical chunking |
| `vector_backend` | Chroma implementation |
| `embedder_backend` | HuggingFace vs other |
| `rerank_enabled` | default rerank on/off |
| `rerank_model` | cross-encoder model name |
| `rerank_candidate_multiplier` | candidate pool size vs `top_k` |
| `hybrid_candidate_multiplier` | RRF candidate pool |
| `sentence_window_size` | only for `sentence_window` chunker |

**Do not** mix these into the index-metadata box.

---

### Tier B — Index metadata (Chroma collection metadata)

Header: **Index metadata** · subtitle: *Chroma collection · one per `index_id`*

| Key | Values / notes |
|-----|----------------|
| `embedding_model` | e.g. `all-MiniLM-L6-v2` — locked on re-ingest |
| `chunker` | `simple` · `hierarchical` · section-based (`markdown`) · `sentence_window` · `semantic` — locked on re-ingest |
| `indexer` | `vector` · `bm25` · `hybrid` — set by `_write_indexer_choice`; locked on re-ingest |
| `description` | optional string (max 500 chars); `write_index_description` |

Muted lines inside box:
- *Written:* `ChromaVectorStore.create_collection` / `modify_metadata`
- *Read:* `GET /indices`, `_read_indexer_choice` on `POST /retrieve`

**Note for `bm25`-only indexes:** Chroma collection may exist with metadata but **zero** vector chunks.

---

### Tier C — Chunk metadata (per `BaseNode` at chunking)

Header: **Chunk metadata** · subtitle: *set at ingest · unchanged through indexing*

Show as **grouped bullets** (not one flat table):

**All chunkers**
- `source` — corpus filename
- `file_type` — `txt` | `pdf`

**PDF only**
- `page` — int

**All chunkers** (`_set_chunk_node_ids`)
- `chunk_role` — `chunk` | `parent` | `child`
- `level` — `0` | `1`

**Hierarchical child only**
- `parent_id` — parent `node_id`

**`sentence_window` only**
- `window` — surrounding sentences (embedded text)
- `original_text` — single sentence

Muted line: *API:* `RetrievedChunk` — `chunk_id`, `text`, `score`, `metadata`

---

### Tier D — Storage copy (footer band, optional but recommended)

Single muted band below Tier C:

| Store | What is stored |
|-------|----------------|
| **Chroma** | per-chunk `metadata` dict on each embedding |
| **Node store** | full node (`node_id`, `text`, `metadata`) for parent nodes |
| **Sparse BM25** | `chunk_id`, `text`, `source`, `metadata` per record |

Same chunk `metadata` dict in all three — no separate BM25-specific fields.

---

## 3. Visual style

| Rule | Detail |
|------|--------|
| **Title** | `Retrieval — metadata structure` only |
| **Layout** | Vertical stack of 3–4 bordered boxes, centered, readable at 16:9 |
| **Typography** | Monospace or clean sans; field names in `code` style |
| **Color** | Light background; tier headers slightly darker; footer muted gray |
| **Arrows** | None required; if used, at most faint ↓ between tiers |
| **Forbidden** | Workflow modules, factories, orchestrator, triad-rag, layout labels |

---

## 4. Copy-paste Gemini prompt

```
Draw a clean technical schema diagram: "Retrieval — metadata structure".

NOT a workflow diagram. No main.py, no factories, no package containers.

Vertical stack of bordered boxes (top to bottom):

1) SERVICE CONFIG (muted subtitle: service-wide, not per index)
   env.toml at startup: sparse_backend, node_store_backend, vector_backend,
   embedder_backend, rerank_enabled, rerank_model, rerank_candidate_multiplier,
   hybrid_candidate_multiplier, sentence_window_size

2) INDEX METADATA (subtitle: Chroma collection, one per index_id)
   Keys: embedding_model, chunker, indexer (vector|bm25|hybrid), description
   Notes inside box: set on first ingest; indexer locked on re-ingest
   Read by GET /indices and POST /retrieve

3) CHUNK METADATA (subtitle: set at chunking, copied unchanged to stores)
   Groups:
   - all: source, file_type
   - pdf: page
   - all: chunk_role, level
   - hierarchical child: parent_id
   - sentence_window: window, original_text
   API note: RetrievedChunk.metadata on POST /retrieve

4) FOOTER BAND (muted): same metadata dict copied to Chroma embeddings,
   node store nodes, sparse BM25 records

Style: light background, clear headers, no flowchart arrows between modules.
Forbidden: triad-rag, COLUMN 1, ROW 1, Stack, A1, orchestrator, main.py boxes.

Output one diagram image.
```

---

## 5. Repair prompt (if Gemini drifts)

```
Redraw as a METADATA SCHEMA only — not the retrieval workflow.

Fix:
1. Remove main.py, factories, indexers/, embedders/, and any HTTP route flow
2. Keep exactly 3 main tiers: service config → index metadata → chunk metadata
3. Index metadata box must list only: embedding_model, chunker, indexer, description
4. Chunk metadata must show conditional groups (pdf, hierarchical, sentence_window)
5. Optional muted footer: Chroma + node store + sparse BM25 all copy the same metadata dict
6. No arrows to modules; no triad-rag or layout grid labels

Title: Retrieval — metadata structure
```

---

## 6. Checklist

- [ ] Title: `Retrieval — metadata structure`
- [ ] Service config tier separate from index metadata (not merged)
- [ ] Index metadata: 4 keys + ingest lock note + `/indices` / `/retrieve` read note
- [ ] Chunk metadata: grouped by chunker type, not one undifferentiated list
- [ ] `RetrievedChunk.metadata` mentioned on chunk tier
- [ ] Footer or note: same dict → Chroma / node store / sparse
- [ ] No workflow modules, no orchestrator, no forbidden layout labels
