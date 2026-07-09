# Orchestrator workflow — Gemini diagram (main + downstream services)

Generate from this spec. The orchestrator has **no factory packages** — it is a thin HTTP edge that loads config and calls **Retrieval** and **Generation** over HTTP.

**Code truth:** `orchestrator/app/main.py` · `orchestrator/app/config.py`

**Authoring note:** Section labels in this doc are for **you only** — never on the generated image.

---

## 1. Mental model

Orchestrator is the **Chat UI entry point** (`:8100`). It does not import retrieval or generation code — only `httpx` calls to their URLs from `env.toml`.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  HEADER: env.toml [orchestrator] ──loads──→ settings (config.py)           │
│          retrieval_url · generation_url · request_timeout_s · retries      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─ main.py (orchestrator · :8100) ─────────────────────────────────────┐  │
│  │ GET /health                                                          │  │
│  │ POST /query  →  retrieve → _chunks_to_context → generate             │  │
│  │ GET /indices · POST /indices/{id}/description  (proxy → retrieval)   │  │
│  │ GET /models · POST /models/select  (proxy → generation)              │  │
│  │ _call_service  (httpx + retry 502/503/504)                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │ reads settings ──→ header                                        │
│         │                                                                  │
│         ├──────── HTTP ────────►  ┌─ Retrieval :8101 ─────────────────┐   │
│         │                         │ POST /retrieve                    │   │
│         │                         │ GET /indices                      │   │
│         │                         │ POST /indices/{id}/description    │   │
│         │                         │ (external service — not in repo)  │   │
│         │                         └───────────────────────────────────┘   │
│         │                                                                  │
│         └──────── HTTP ────────►  ┌─ Generation :8102 ─────────────────┐   │
│                                   │ POST /generate                    │   │
│                                   │ GET /models                       │   │
│                                   │ POST /models/select               │   │
│                                   │ (external service — not in repo)  │   │
│                                   └───────────────────────────────────┘   │
│                                                                            │
│  muted (optional): Chat UI ──POST /query──► main.py                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### Diagram type

- **Coordination / integration** — not a package-factory diagram like retrieval or generation.
- **Center or left:** tall `main.py` box (orchestrator routes + helpers).
- **Right or below:** two **bordered external service containers** — Retrieval and Generation — each lists **endpoints only**, not internal classes.
- **Header above:** `env.toml` → `settings`.

### Rules

1. **Title:** `Orchestrator — main.py workflow` only (no triad-rag, no PRD labels)
2. **Do not** draw retrieval factories (indexers, chunkers, …) or generation `ai_providers/` — those belong in their own diagrams
3. **Do not** draw arrows from Retrieval to Generation — orchestrator calls each independently
4. **No** `COLUMN 1`, `ROW 1`, `Stack`, `A1`, layout grid labels on the image
5. Label external boxes **Retrieval :8101** and **Generation :8102** (ports optional but helpful)

---

## 2. Module inventory

| # | Box | Inside |
|---|-----|--------|
| H | **config header** | `env.toml` `[orchestrator]` · `settings` (`config.py`) |
| 1 | **`main.py`** | HTTP routes + `_chunks_to_context` · `_call_service` (§2.1) |
| 2 | **Retrieval :8101** (external) | Downstream endpoints orchestrator calls (§2.2) |
| 3 | **Generation :8102** (external) | Downstream endpoints orchestrator calls (§2.3) |

---

## 2.1 `main.py (API entry · :8100)`

```
GET /health

POST /query          ← primary Q&A flow (Chat UI)
GET /indices
POST /indices/{index_id}/description
GET /models
POST /models/select
```

| Route | What it does |
|-------|----------------|
| `POST /query` | `POST` retrieval `/retrieve` → map `chunks` to `sources` → `_chunks_to_context` → `POST` generation `/generate` → `QueryResponse` |
| `GET /indices` | Proxy `GET` retrieval `/indices` |
| `POST /indices/{id}/description` | Proxy `POST` retrieval `/indices/{id}/description` |
| `GET /models` | Proxy `GET` generation `/models` |
| `POST /models/select` | Proxy `POST` generation `/models/select` |

**`POST /query` body:** `question`, `top_k`, `index_id`, optional `rerank` (forwarded to retrieval).

**`POST /query` response:** `answer`, `sources` (from retrieval `chunks`; no `candidates`).

**Helper note (inside main.py box):** `_chunks_to_context` — numbered passage text `[1] …`, `[2] …`, max ~12000 chars → generation `context`.

**Helper note:** `_call_service` — `httpx` async client; retries on 502/503/504 per `retry_attempts` / `retry_wait_s`.

---

## 2.2 External container: Retrieval `:8101`

Muted subtitle: *separate service — see retrieval workflow diagram*

| Endpoint | Called from orchestrator |
|----------|-------------------------|
| `POST /retrieve` | `POST /query` |
| `GET /indices` | `GET /indices` |
| `POST /indices/{index_id}/description` | `POST /indices/{id}/description` |

Do **not** show Chroma, BM25, chunkers, or ingest inside this box.

---

## 2.3 External container: Generation `:8102`

Muted subtitle: *separate service — see generation workflow diagram*

| Endpoint | Called from orchestrator |
|----------|-------------------------|
| `POST /generate` | `POST /query` (after context built) |
| `GET /models` | `GET /models` |
| `POST /models/select` | `POST /models/select` |

Do **not** show `ai_providers/`, Gemini, or `prompts.toml` inside this box.

---

## 2.4 Config header (required)

| Box | Content |
|-----|---------|
| **env.toml** | `[orchestrator]` — `retrieval_url`, `generation_url`, `request_timeout_s`, `retry_attempts`, `retry_wait_s` |
| **settings** | `config.py` — same fields loaded at startup (`ORCH_` env prefix optional override) |

No `prompts.toml` on orchestrator. No `[retrieval]` or `[generation]` tables in this header — only `[orchestrator]`.

---

## 2.5 `POST /query` sequence (box text on main.py or small muted inset)

Show as **numbered steps inside** `main.py` (not separate arrows):

```
1. POST /retrieve  (query, top_k, index_id, rerank)
2. chunks → sources + _chunks_to_context
3. POST /generate  (question, context)
4. answer + sources → client
```

If no hits: skip step 3; return empty-sources message (note in small text).

---

## 3. Arrows — simple

**Hit box borders only** — settings header, Retrieval container, Generation container.

| # | From | To | Label |
|---|------|-----|-------|
| 1 | `env.toml` | `settings` | `loads` |
| 2 | `main.py` | `settings` (header) | `reads settings` |
| 3 | `main.py` | **Retrieval :8101** container | `retrieval` |
| 4 | `main.py` | **Generation :8102** container | `generation` |

**Total: 4 labeled arrows.** Do not add per-endpoint arrows.

### Optional UIs (muted, dashed — omit if crowded)

| UI | Calls | Label |
|----|-------|-------|
| `ui/chat.py` | orchestrator `POST /query` | `POST /query` |
| `retrieval/ui/index.py` | retrieval `:8101` directly (ingest, query) | *(no arrow to orchestrator)* |
| `eval/ui/run.py` | retrieval + generation directly | *(no arrow to orchestrator)* |

### Do NOT draw

- Retrieval → Generation (no direct link)
- Arrows into inner endpoint lines inside external containers
- Index UI (`retrieval/ui/index.py`) calls retrieval directly, not orchestrator
- `triad-rag`, `COLUMN 1`, `ROW 1`, `Stack`, `A1`

**Legend (tiny):** `—— HTTP` · `- - - optional caller`

---

## 4. Gemini master prompt (copy-paste)

```
Draw a clean technical architecture diagram: "Orchestrator — main.py workflow".

INTEGRATION diagram — not retrieval/generation internals.

Layout:
- TOP HEADER: env.toml [orchestrator] → settings (config.py)
  Fields: retrieval_url, generation_url, request_timeout_s, retry_attempts, retry_wait_s

- LEFT or CENTER (tall box): main.py (:8100)
  Routes: GET /health, POST /query, GET /indices, POST /indices/{id}/description,
          GET /models, POST /models/select
  Helpers: _call_service (httpx + retry), _chunks_to_context
  POST /query sequence (numbered inside box):
    1. POST /retrieve  2. chunks → context  3. POST /generate  4. answer + sources

- RIGHT or BELOW: two BORDERED external service containers (not internal packages):

  Retrieval :8101 (external service)
    POST /retrieve, GET /indices, POST /indices/{id}/description
    subtitle: separate service — no indexers/chunkers inside

  Generation :8102 (external service)
    POST /generate, GET /models, POST /models/select
    subtitle: separate service — no ai_providers inside

ARROWS — exactly 4:
1. env.toml → settings: "loads"
2. main.py → settings: "reads settings"
3. main.py → Retrieval container border: "retrieval"
4. main.py → Generation container border: "generation"

NO arrow from Retrieval to Generation.
NO factory packages inside orchestrator.
Optional muted dashed: ui/chat.py → main.py "POST /query"
Note (text only, no arrow): retrieval/ui/index.py and eval/ui/run.py call retrieval/generation directly

Forbidden: triad-rag, PRD labels, COLUMN 1, ROW 1, Stack, A1, retrieval internal modules, generation internal modules.

Output one diagram image.
```

---

## 5. Repair prompt (if Gemini drifts)

```
Redraw as ORCHESTRATOR INTEGRATION only.

Fix:
1. Remove indexers, chunkers, embedders, ai_providers, factories — those are other services
2. Keep exactly: config header + main.py + Retrieval :8101 box + Generation :8102 box
3. Retrieval and Generation boxes list HTTP endpoints only — mark as external services
4. Exactly 4 arrows: loads, reads settings, retrieval, generation (hit container borders)
5. No Retrieval → Generation arrow; orchestrator calls each separately
6. POST /query flow is numbered steps INSIDE main.py, not a spaghetti of extra arrows
7. Remove COLUMN 1, ROW 1, triad-rag, Stack, A1

Title: Orchestrator — main.py workflow
```

---

## 6. Checklist

- [ ] Title: `Orchestrator — main.py workflow`
- [ ] Config header: `env.toml` `[orchestrator]` only → `settings`
- [ ] `main.py` lists all 6 routes + `_call_service` + `_chunks_to_context`
- [ ] `POST /query` 4-step sequence visible on main.py
- [ ] Retrieval :8101 external box — 3 endpoints, no internal packages
- [ ] Generation :8102 external box — 3 endpoints, no internal packages
- [ ] Exactly 4 arrows (loads · reads settings · retrieval · generation)
- [ ] No Retrieval → Generation arrow
- [ ] No triad-rag / layout chrome / forbidden labels
