# triad-rag — design

Design for all three services: orchestrator, retrieval, and generation. Each section below has a diagram and a short description of what happens.

Config: `env.toml` at repo root. LLM system prompt: `generation/prompts.toml`.

---

## Rules

- Three services — separate processes, **HTTP + JSON only** (no cross-imports).
- **Settings** read in each service’s `main.py` and `config.py` only. Other code gets plain values and instances.
- **Factories** (`make_chunker`, `make_provider`, …) called from `main.py`, not from deep packages.
- **Ingester UI** → retrieval only. **Chat UI** → orchestrator only.

---

## System overview

Two Streamlit apps, three backend services:

| What | Role | Calls |
|------|------|-------|
| **Ingester UI** (`retrieval/ingester_ui.py`) | Upload files, manage collections, test search (passages only) | **Retrieval** `:8101` |
| **Chat UI** (`app_ui.py`) | Ask questions; get answers + sources | **Orchestrator** `:8100` |
| **Retrieval** | Store files, build indexes, find passages | — |
| **Orchestrator** | Chains retrieval + generation for Q&A | Retrieval `:8101`, Generation `:8102` |
| **Generation** | Writes an answer from question + passages | External LLM API |

**Upload:** Ingester UI → retrieval → files saved and indexed.

**Chat:** Chat UI → orchestrator → retrieval (passages) → generation (answer) → Chat UI.

![System overview](diagrams/system-overview.png)

---

## Two UIs

Different apps, run separately, for different jobs.

### Ingester UI — upload and test search

**File:** `retrieval/ingester_ui.py` · **Run:** `streamlit run ingester_ui.py` (from `retrieval/`) · **Talks to:** `http://localhost:8101`

| Tab / area | What you do | Retrieval endpoint |
|------------|-------------|-------------------|
| **Ingest** | Upload PDF/txt, pick indexer/chunker/model (new collection) | `POST /ingest` |
| **Manage** | List collections, delete files/index | `GET /indices`, `DELETE …` |
| **Retrieve** | Test search — passages and scores, no LLM | `POST /retrieve` |

Use it to upload files, manage collections, and run search **without** an LLM answer — so you can see passages, scores, and (with rerank on) the wider hit list.

```
Ingester UI  ──────────────────────────────►  Retrieval :8101
              (upload · manage · search)
```

### Chat UI — ask questions, get answers

**File:** `app_ui.py` · **Run:** `streamlit run app_ui.py` (from `triad-rag/`) · **Talks to:** `http://localhost:8100`

| What you do | You call (orchestrator) | Orchestrator then calls |
|-------------|-------------------------|-------------------------|
| Ask a question | `POST /query` | retrieval `POST /retrieve`, then generation `POST /generate` |
| Pick collection or model | `GET /indices`, `GET /models`, `POST /models/select` | retrieval or generation |

The Chat UI never calls retrieval or generation directly. Orchestrator does that for you.

```
Chat UI  ──POST /query──►  Orchestrator :8100
                                ├──► Retrieval :8101   (passages)
                                └──► Generation :8102  (answer)
                           ◄── answer + sources ──
```

| | Ingester UI | Chat UI |
|---|-------------|---------|
| **Job** | Upload files; test search | Ask questions; get answers |
| **Talks to** | Retrieval `:8101` | Orchestrator `:8100` |
| **Search result** | Passage list | Answer + source passages |
| **Extra hit list (rerank)?** | Yes | No |
| **Uses LLM?** | No | Yes |

---

## Technical architecture

Services are independent processes. Each has `main.py` as the HTTP entry point and small packages for pluggable parts (chunkers, providers, etc.). Factories in each service create the right implementation from config.

![Technical architecture](diagrams/technical_architecture.png)

### Repo layout

```
triad-rag/
├── env.toml                 # shared config ([retrieval], [generation], [orchestrator])
├── app_ui.py                # Chat UI → orchestrator
├── orchestrator/
│   └── app/
│       ├── config.py
│       └── main.py          # POST /query; calls retrieval + generation
├── retrieval/
│   ├── ingester_ui.py       # Ingester UI → retrieval
│   ├── data/
│   │   ├── corpus/          # uploaded .txt / .pdf files
│   │   └── index_store/     # Chroma, BM25, node stores
│   └── app/
│       ├── config.py
│       ├── main.py          # ingest + retrieve APIs
│       ├── chunkers/
│       ├── embedders/
│       ├── indexers/
│       ├── stores/
│       ├── rerankers/
│       ├── hybrid/
│       └── ingest/
└── generation/
    ├── prompts.toml         # system prompt for the model
    └── app/
        ├── config.py
        ├── main.py          # generate + model selection APIs
        └── ai_providers/
```

---

## APIs

### Orchestrator (`:8100`) — Chat UI only

| Method | Path | What it does |
|--------|------|----------------|
| GET | `/health` | Health check |
| POST | `/query` | Search retrieval, then ask generation; return answer + sources |
| GET | `/indices` | List collections (orchestrator calls retrieval) |
| GET | `/models` | List models (orchestrator calls generation) |
| POST | `/models/select` | Change model (orchestrator calls generation) |
| POST | `/indices/{id}/description` | Set description (orchestrator calls retrieval) |

### Retrieval (`:8101`) — Ingester UI (and orchestrator for chat)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/indices` | List collections and their settings |
| GET | `/ingest/options` | Valid chunkers, models, indexers for upload |
| POST | `/ingest` | Upload a file and index it |
| POST | `/retrieve` | Search: return top passages for a question |
| POST | `/indices/{id}/description` | Update collection description |
| DELETE | `/indices/{id}` | Delete a collection |
| GET | `/indices/{id}/files` | List files in a collection |
| DELETE | `/indices/{id}/corpus` | Clear all files in a collection |
| DELETE | `/indices/{id}/files/{name}` | Delete one file |

**`POST /retrieve`:** send `query`, `top_k`, `index_id`, optional `rerank`. Returns passages in `chunks` (and optional `candidates` when rerank builds a wider list). Ingester UI uses this directly; orchestrator uses it during chat and only keeps `chunks`.

**`POST /ingest`:** upload a file (form fields). Ingester UI only.

### Generation (`:8102`) — answer from context

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/models` | List providers and model aliases |
| POST | `/models/select` | Switch provider / model |
| POST | `/generate` | Answer a question using provided context |

**`POST /generate` body:** `question`, `context` (passages as plain text). Generation does not call retrieval itself.

---

## Orchestrator workflow

Chat UI talks here (`:8100`). `main.py` reads `env.toml` `[orchestrator]` and calls retrieval and generation over HTTP.

**`POST /query`:** `POST /retrieve` on retrieval → turn passages into numbered text → `POST /generate` on generation → return `answer` + `sources`.

**Other routes:** indices and descriptions go to retrieval; models go to generation. Chat UI only needs `:8100`.

![Orchestrator workflow](diagrams/orchestrator_main.png)

### Files

```
orchestrator/app/
├── main.py       # routes, HTTP client, builds context from passages
└── config.py     # retrieval_url, generation_url, timeout, retries
```

| Service | Port | Called for |
|---------|------|------------|
| Retrieval | `:8101` | `/retrieve`, `/indices`, `/indices/{id}/description` |
| Generation | `:8102` | `/generate`, `/models`, `/models/select` |

---

## Retrieval workflow

Ingester UI talks here directly. Orchestrator calls here when you chat.

`main.py` loads settings, runs factories, and wires HTTP routes to packages. It is the **composition root** — the only place that reads config and calls `make_*` factories.

**Ingest:** save file → chunk → embed (if vector/hybrid) → write to stores → save index settings.

**Retrieve:** load index → search by stored mode (vector, keyword, or both) → optional rerank → return passages.

![Retrieval workflow](diagrams/retrieval_main_workflow.png)

### Packages

```
retrieval/app/
├── main.py              # all HTTP routes; _search pipeline
├── config.py            # settings from env.toml [retrieval]
├── chunkers/            # split files into passages
├── embedders/           # text → vectors (HuggingFace)
├── indexers/            # ChromaIndexer (vector), Bm25Indexer (keyword)
├── stores/              # Chroma vector, node store, BM25 sparse
├── rerankers/           # optional second-pass scoring
├── hybrid/              # merge vector + keyword results (RRF)
└── ingest/              # file upload helpers
```

### Three storage roles

| Store | Holds | Search use | On disk |
|-------|-------|------------|---------|
| **Vector** | Embeddings + vectors | Meaning similarity | `index_store/chroma/` |
| **Node** | Non-embedded nodes (e.g. parents) | Auto-merge to wider passages | `index_store/node_store/{id}.json` or `.sqlite` |
| **Sparse** | Chunk text for BM25 | Keyword search | `index_store/sparse/<id>/` |

### Search mode vs rerank

**Search mode** (picked once at first upload, stored on the collection):

| Mode | Meaning |
|------|---------|
| `vector` | Similar meaning to the question |
| `bm25` | Matching keywords |
| `hybrid` | Both, then combined (rank fusion) |
| rerank (optional) | Re-order a wider hit list for better results |

**Order inside `_search`:**

```
bm25:     BM25 search → optional rerank
hybrid:   vector search → BM25 search → fusion → optional rerank
vector:   vector search → optional rerank
```

Fusion merges two hit lists (no model). Rerank re-scores one candidate list (cross-encoder). They are different optional stages.

### What each UI controls

| UI | Talks to | At upload | At search |
|----|----------|-----------|-----------|
| **Ingester UI** | Retrieval `:8101` | Indexer, chunker, embedding model | Rerank checkbox only — search mode is fixed per collection |
| **Chat UI** | Orchestrator `:8100` | Collection, model | Rerank checkbox (passed to retrieval) |

Search mode is not chosen per question. It is stored on the collection when you first upload.

### Example API flows

Who calls what, what JSON comes back, what the UI shows. Examples are shortened.

#### Call map

| You do this | UI | HTTP call | Who returns the JSON |
|-------------|-----|-----------|----------------------|
| Upload a file | Ingester UI | `POST http://localhost:8101/ingest` | **Retrieval** |
| Test search | Ingester UI | `POST http://localhost:8101/retrieve` | **Retrieval** |
| Ask a question | Chat UI | `POST http://localhost:8100/query` | **Orchestrator** (after retrieval + generation) |
| List collections (ingester) | Ingester UI | `GET http://localhost:8101/indices` | **Retrieval** |
| List collections (chat) | Chat UI | `GET http://localhost:8100/indices` | **Orchestrator** (asks retrieval) |

#### 1. Upload a file

```
Ingester UI  ──POST /ingest──►  Retrieval :8101  ──►  IngestResponse
```

**Retrieval returns:**

```json
{
  "index_id": "default",
  "saved_as": "policy.pdf",
  "chunks_indexed": 42,
  "embedding_model": "all-MiniLM-L6-v2",
  "chunker": "simple",
  "indexer": "hybrid",
  "ready": true
}
```

| `indexer` at upload | What retrieval built | `embedding_model` |
|---------------------|----------------------|-------------------|
| `vector` | Chroma embeddings only | model name |
| `bm25` | BM25 keyword index only | `null` |
| `hybrid` | Both vector and BM25 | model name |

#### 2. Test search (Ingester UI)

```
Ingester UI  ──POST /retrieve──►  Retrieval :8101  ──►  RetrieveResponse
```

You do **not** send search mode — retrieval reads `indexer` from collection metadata.

```json
{
  "query": "What is the refund policy?",
  "index_id": "default",
  "chunks": [
    {
      "chunk_id": "policy.pdf#p2chunk1",
      "text": "Refunds are available within 30 days…",
      "score": 0.82,
      "metadata": { "source": "policy.pdf", "file_type": "pdf", "page": 2 }
    }
  ],
  "candidate_count": 0,
  "candidates": []
}
```

With **rerank on**, `chunks` is the reranked top-k and `candidates` holds the wider pre-rerank pool (Ingester UI shows both).

#### 3. Ask a question (Chat UI)

```
Chat UI  ──POST /query──►  Orchestrator :8100
                              ├── POST /retrieve  → Retrieval
                              └── POST /generate  → Generation
                           ◄── answer + sources
```

**Inside orchestrator:**

1. Call retrieval `POST /retrieve`
2. Take `chunks` (ignore `candidates`)
3. Build numbered passage text for the model (~12k char cap)
4. Call generation `POST /generate`
5. Return `answer` + `sources` to Chat UI

**Orchestrator returns** (only JSON the Chat UI sees):

```json
{
  "answer": "Refunds are available within 30 days of purchase…",
  "sources": [
    {
      "chunk_id": "policy.pdf#p2chunk1",
      "text": "Refunds are available within 30 days…",
      "score": 0.82,
      "metadata": { "source": "policy.pdf", "file_type": "pdf", "page": 2 }
    }
  ]
}
```

`sources` are the same passages as retrieval’s `chunks`. No `candidates` in this response.

---

## Generation workflow

Used when you chat — orchestrator calls `POST /generate` after search. Ingester UI does not use this service.

`main.py` loads settings, picks an LLM provider, calls `generate(question, context)`.

![Generation workflow](diagrams/generation_main_workflow.png)

### Packages

```
generation/app/
├── main.py              # HTTP routes
├── config.py            # settings from env.toml + prompts.toml
└── ai_providers/
    ├── provider_factory.py
    ├── gemini_provider.py   # Google Gemini API
    └── stub_provider.py     # local fake answers (dev/test)
```

Config sources:
- `env.toml` `[generation]` — provider, model, temperature, API keys
- `prompts.toml` — `system_prompt` (loaded into `settings.system_prompt` at startup)

---

## Metadata structure

Three levels of metadata in retrieval:

1. **Service config** (`env.toml`) — global defaults: backends, rerank settings, allowed chunkers/models. Not stored per collection.

2. **Index metadata** (one record per collection in Chroma) — `embedding_model`, `chunker`, `indexer`, `description`. Set on first ingest; `indexer` is locked on re-ingest.

3. **Chunk metadata** (per passage) — `source`, `file_type`, optional `page` (PDF), `chunk_role` / `parent_id` (hierarchical), `window` / `original_text` (sentence window). Same dict is copied to vector store, node store, and BM25 store. Returned on `POST /retrieve` as `RetrievedChunk.metadata`.

![Metadata structure](diagrams/retrieval_metadata_structure.png)

---

## Related docs

| Doc | Contents |
|-----|----------|
| [`../README.md`](../README.md) | Getting started, commands |
| [`chunking-strategies.md`](chunking-strategies.md) | How each chunker splits documents |
| [`retrieval-strategies.md`](retrieval-strategies.md) | Vector, BM25, hybrid, rerank |
| [`diagrams/markdowns/`](diagrams/markdowns/) | Regenerate workflow images |
