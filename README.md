# triad-rag

Upload documents, search them, ask questions, get answers from your files.

Three services (retrieval, orchestrator, generation) and two UIs. They talk over HTTP only — no shared Python code between services.

![System overview](docs/diagrams/system-overview.png)

| | Talks to |
|---|----------|
| **Ingester UI** (`retrieval/ingester_ui.py`) | Retrieval `:8101` — upload, manage collections, test search |
| **Chat UI** (`app_ui.py`) | Orchestrator `:8100` — questions and answers |
| **Orchestrator** `:8100` | Retrieval + generation (for chat and settings) |
| **Retrieval** `:8101` | Files, indexes, search |
| **Generation** `:8102` | LLM answers from passages |

---

## How it works

**Upload:** Ingester UI → retrieval — save files, chunk, build indexes (`corpus/` + `index_store/`).

**Chat:** Chat UI → orchestrator → retrieval (passages) → generation (answer).

Search mode is chosen once per collection at upload (`vector`, `bm25`, or `hybrid`). Optional rerank is per search. Retrieval never calls the LLM; generation never searches files.

**Design details:** modules, routes, disk layout, composition rules → [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Capabilities & caveats

What works today and what to watch for when picking upload and search options. Details: [`docs/chunking-strategies.md`](docs/chunking-strategies.md), [`docs/retrieval-strategies.md`](docs/retrieval-strategies.md).

### Files & ingest

- **Upload types:** `.txt` and `.pdf` only.
- **PDF reading:** `pypdf` extracts **flat text per page** — no headings, tables, or layout structure.
- **Per collection, locked on re-ingest:** chunker, indexer (`vector` / `bm25` / `hybrid`), and embedding model. Create a new `index_id` to change them.
- **Re-upload** replaces that file’s chunks in the same collection; it does not change the locked settings above.

### Chunkers (`chunker_name` / Ingester **Chunker**)

| Chunker | Use when | Chunker-only caveats |
|---------|----------|----------------------|
| **simple** (default) | General PDF/txt | Fixed-size splits; no document structure. |
| **section-based** (`markdown`) | Markdown with `#` headings | **Markdown text only.** Convert PDFs to markdown **before** upload, then ingest as `.txt`. Raw `.pdf` has no headings — section splitting will not work. |
| **hierarchical** | Long docs: small hits at search, wider parent passage when useful | Parents are stored for lookup, not searched directly — see [chunker × indexer](#chunker--indexer) below. |
| **sentence_window** | FAQ-style Q&A | Search matches a **window** of nearby sentences; the single sentence lives in metadata. |
| **semantic** | Topics change within a page | **Needs the embedding model at ingest** to find topic boundaries (same model as vector search). |

### Search (indexer)

- **Indexer is fixed per collection** — queries always use what was chosen at first upload (not a per-query dropdown).
- **`vector`:** meaning search via embeddings; **parent merge** runs here when the chunker keeps parents (hierarchical).
- **`bm25`:** keyword search on `embed_chunks` only; needs `sparse_backend` = `json_bm25` or `sqlite_bm25` in `env.toml` (`none` disables bm25/hybrid). **No parent merge** — results are always the indexed child/window/section text.
- **`hybrid`:** vector + BM25, merged with RRF; parent merge applies on the **vector leg only**; optional rerank per query.

### Chunker × indexer

Every indexer searches **`embed_chunks`** — the passages actually written to Chroma and/or BM25. Some chunkers also keep extra nodes (e.g. hierarchical **parents**) in a node store; those are used only when **vector** search can **auto-merge** several child hits into one parent passage (`ratio > 0.4`). BM25 never reads the node store.

| Chunker | `vector` | `bm25` | `hybrid` |
|---------|----------|--------|----------|
| **simple** | Works well — default for most PDF/txt | Works well — keyword on same fixed-size chunks | Works well — general-purpose |
| **section-based** (`markdown`) | Works well — **if** file is markdown `.txt` with `#` headings | Same, keyword-only — still requires markdown text | Works well for structured markdown |
| **hierarchical** | **Best fit** — child hits + optional parent text back | **Limited** — only **child** chunks are indexed; parents are **not** stored on bm25-only ingest, so you get small keyword hits with **no** parent expansion. Prefer **simple** for bm25-only, or use **hybrid** / **vector** | **Good** — parent merge on the vector leg; BM25 leg still returns child text only |
| **sentence_window** | Works well — meaning match on sentence windows | Works well — keyword match on window text | Works well for FAQ-style content |
| **semantic** | Works well — topic-sized chunks | Works, but embeddings are used only to **cut** chunks at ingest; search stays keyword-only | Works well — combines topic chunks with keywords |

**Practical picks**

- **Most PDF/txt, unsure:** `simple` + `hybrid` (or `vector` if BM25 is off).
- **Markdown with headings:** `markdown` + `vector` or `hybrid` (upload as `.txt`).
- **Long docs, want wider context in results:** `hierarchical` + `vector` or `hybrid` — **not** `bm25` alone.
- **Keyword-heavy, no embeddings:** `simple` + `bm25`.
- **FAQ / short factual lines:** `sentence_window` + any indexer.

On **bm25-only** collections the UI may still ask for an embedding model — that value is stored in collection metadata and is required for **semantic** chunking; it is **not** used to build the BM25 index.

### Generation (chat)

- Needs a configured provider in `env.toml` (`google`, `openai`, `anthropic`, or **`stub`** for local testing without an API).
- Answers use **retrieved passages only** — quality depends on chunking, indexer, and source files.

---

## Getting started

### 1. Setup (once)

```bash
cd triad-rag
python3 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt
cp env.toml.bak env.toml
```

Edit `env.toml`: set `api_key` under `[generation.google]` (or use `stub` for local testing), and fill in `corpus_dir` / `index_store_dir` under `[retrieval]` if empty.

First upload may download the embedding model (`all-MiniLM-L6-v2`). Retrieval loads all models in `available_embedding_models` at startup.

### 2. Run the three services

Three terminals. Activate the venv in each. Run from the service folder.

```bash
source .venv313/bin/activate

# Terminal 1 — orchestrator :8100
cd orchestrator && python -m app.main

# Terminal 2 — retrieval :8101
cd retrieval && python -m app.main

# Terminal 3 — generation :8102
cd generation && python -m app.main
```

Optional: `--host 0.0.0.0`, `--no-reload`.

```bash
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1:8101/health
curl -s http://127.0.0.1:8102/health
```

### 3. Run the UIs

**Ingester** (upload + test search):

```bash
cd retrieval
streamlit run ingester_ui.py
```

**Chat** (questions + answers):

```bash
cd triad-rag
streamlit run app_ui.py
```

```bash
export RETRIEVAL_API_URL=http://127.0.0.1:8101
export ORCHESTRATOR_API_URL=http://127.0.0.1:8100
```

### 4. First use

1. **Ingester UI** — upload a file, pick indexer/chunker on a new collection.
2. Same UI — **Query** tab to test search (passages only).
3. **Chat UI** — pick collection, ask a question.

Or use [Commands](#commands) below.

---

## Commands

| Task | Command |
|------|---------|
| List collections | `curl -s http://127.0.0.1:8101/indices` |
| Ingest options | `curl -s http://127.0.0.1:8101/ingest/options` |
| Upload file | `curl -s -X POST http://127.0.0.1:8101/ingest -F file=@doc.pdf -F index_id=default -F indexer=hybrid -F embedding_model=all-MiniLM-L6-v2 -F chunker_name=simple` |
| Search (passages) | `curl -s -X POST http://127.0.0.1:8101/retrieve -H 'Content-Type: application/json' -d '{"query":"What is RAG?","top_k":3,"index_id":"default","rerank":false}'` |
| Chat (answer + sources) | `curl -s -X POST http://127.0.0.1:8100/query -H 'Content-Type: application/json' -d '{"question":"What is RAG?","index_id":"default","top_k":3,"rerank":false}'` |
| List models | `curl -s http://127.0.0.1:8102/models` |
| Run eval | `python eval/run_eval.py` (repo root, all services up) |

API docs: `http://127.0.0.1:8100/docs`, `:8101/docs`, `:8102/docs`.

---

## Configuration

`env.toml.bak` → `env.toml` (gitignored).

| Section | Main settings |
|---------|----------------|
| `[orchestrator]` | `retrieval_url`, `generation_url`, timeouts, retries |
| `[retrieval]` | paths, chunkers, embedding models, `sparse_backend`, rerank |
| `[generation]` | `default_provider`, `temperature` |
| `[generation.google]` | `api_key`, model aliases |

`generation/prompts.toml` — system prompt for the LLM.

Env overrides: `ORCH_*`, `RET_*`, `GEN_*`.

`sparse_backend = "none"` → only `vector` at upload. Use `json_bm25` or `sqlite_bm25` for `bm25` and `hybrid`.

---

## Documentation

### Guides

| Doc | Contents |
|-----|----------|
| [`docs/DESIGN.md`](docs/DESIGN.md) | All three services — orchestrator, retrieval, generation |
| [`docs/chunking-strategies.md`](docs/chunking-strategies.md) | How each chunker splits documents |
| [`docs/retrieval-strategies.md`](docs/retrieval-strategies.md) | Vector, BM25, hybrid, rerank |

### Workflow diagrams

| | File |
|---|------|
| System (above) | [`system-overview.png`](docs/diagrams/system-overview.png) |
| Orchestrator | [`orchestrator_main.png`](docs/diagrams/orchestrator_main.png) |
| Retrieval | [`retrieval_main_workflow.png`](docs/diagrams/retrieval_main_workflow.png) |
| Generation | [`generation_main_workflow.png`](docs/diagrams/generation_main_workflow.png) |
| Metadata | [`retrieval_metadata_structure.png`](docs/diagrams/retrieval_metadata_structure.png) |
| Technical layout | [`technical_architecture.png`](docs/diagrams/technical_architecture.png) |

Regenerate diagrams: [`docs/diagrams/markdowns/`](docs/diagrams/markdowns/).
