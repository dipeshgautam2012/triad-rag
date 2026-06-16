# Generation workflow — Gemini diagram (main + package containers)

Generate from this spec. Mirror the **retrieval** diagram style ([`retrieval-workflow-gemini.md`](retrieval-workflow-gemini.md)): bordered package containers, sparse arrows, no layout chrome.

**API truth:** [`generation-workflow-surface.md`](generation-workflow-surface.md)

**Authoring note:** “Container 1”, “Col 1”, etc. in this doc are for **you only** — never on the generated image.

---

## 1. Mental model

Generation is **much smaller** than retrieval: one factory package (`ai_providers/`) plus config.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADER: env.toml [generation] + [generation.*]   prompts.toml          │
│          ──loads──→ settings (config.py)   code: _PROVIDER_REGISTRY     │
├──────────────┬──────────────────────────────────────────────────────────┤
│              │  ┌─ ai_providers/ ─────────────────────────────────────┐ │
│  main.py     │  │ provider_factory │ BaseAIProvider ┊ GeminiProvider  │ │
│  API entry   │  │ make_provider    │       ┊ dashed   StubProvider     │ │
│              │  └────────────────────────────────────────────────────┘ │
│  ──→ ai_providers/                                                    │
│  reads settings ──→ header                                              │
└──────────────┴──────────────────────────────────────────────────────────┘
```

### Package container pattern

Same rules as retrieval sketch:

1. **Bordered container** with header bar (`ai_providers/`)
2. **Inside:** factory left (`provider_factory`, `make_provider`) | base + derived right
3. **Base** on top → **dashed vertical** ↓ → implementation box(es) below
4. **No** `creates` arrows inside container
5. **No** `COLUMN 1`, `ROW 1`, `triad-rag`, `Stack`, `A1` on the image

### Rules

1. **Left:** tall `main.py` box
2. **Right:** **one** package container (`ai_providers/`) — generation has only one factory module
3. **Header above:** config band (env.toml + prompts.toml → settings)
4. **Title:** `Generation — main.py workflow` only (no triad-rag)
5. **Arrows:** see §3 — hit **container borders**, not inner classes

---

## 2. Module inventory

| # | Container / band | Inside |
|---|------------------|--------|
| H | **config header** | `env.toml` `[generation]` + nested `[generation.google]` / `[generation.stub]` / … · `prompts.toml` · `settings` · registry note `_PROVIDER_REGISTRY` |
| 1 | **`ai_providers/`** | `provider_factory` \| `BaseAIProvider` ┊ `GeminiProvider` · `StubProvider` |

**Left column:** `main.py` — routes + helpers (§2.1)

---

## 2.1 `main.py (API entry)`

```
GET /health
GET /models
POST /models/select
POST /generate

Request body: question + context (plain text)
```

| Route | What it does |
|-------|----------------|
| `GET /models` | `list_models()` — active provider/alias + catalog |
| `POST /models/select` | `set_model(provider, model_alias)` — runtime switch |
| `POST /generate` | `make_provider(settings).generate(question, context)` → `answer` |

**Input note (box text):** `POST /generate` accepts `question` and `context` strings. Generation does not fetch or build context itself.

---

## 2.2 Config header (required)

| Box | Content |
|-----|---------|
| **env.toml** | `[generation]` — `default_provider`, `temperature`, optional token limits |
| **env.toml nested** | `[generation.google]`, `[generation.stub]`, … — `api_key`, `default_model`, `[generation.*.models]` aliases |
| **prompts.toml** | `system_prompt` — model instructions (separate file) |
| **settings** | `config.py` — active `provider`, `model_alias`, `model`, `api_key`, `temperature`, `system_prompt` |
| **registry** (small) | `_PROVIDER_REGISTRY`: `google` → GeminiProvider, `stub` → StubProvider |

**Implemented today:** `google`, `stub`. TOML may list `openai` / `anthropic` but `provider_implemented` is false until a class is added.

---

## 2.3 Container: `ai_providers/`

**Left — provider_factory**
- `make_provider(cfg)`
- `provider_implemented(name)`
- `_PROVIDER_REGISTRY`

**Right — base + derived**
- **BaseAIProvider**
  - `_system_instruction()` — from `settings.system_prompt`
  - `_user_content(question, context)` — formats Context + Question blocks
  - `generate(question, context, output_schema=None)`
- dashed ↓
- Implementation box:
  - **GeminiProvider** — `generate` → `google.genai` API (`generate_content`)
  - **StubProvider** — `generate` → fake local answer (no API)

---

## 2.4 Prompt & context (box text only — no extra arrows)

| Piece | Where set | Where used |
|-------|-----------|------------|
| **System prompt** | `prompts.toml` → `settings.system_prompt` | `BaseAIProvider._system_instruction()` → Gemini `system_instruction` |
| **User message** | `POST /generate` body | `_user_content(question, context)` → model `contents` |
| **Context** | `context` field on request | Formatted inside `_user_content`; optional `[1]`, `[2]` labels in text |

---

## 3. Arrows — simple

**Hit container borders or settings box only.**

| # | From | To | Label |
|---|------|-----|-------|
| 1 | `env.toml` | `settings` | `loads` |
| 2 | `prompts.toml` | `settings` | `system prompt` |
| 3 | `main.py` | `settings` (header) | `reads settings` |
| 4 | `main.py` | **`ai_providers/` container** | `ai_providers` |

**Total: 4 labeled arrows.** Do not add more.

### Structural lines only

- Dashed vertical **BaseAIProvider → implementations** inside `ai_providers/` container
- No `creates` arrows
- No arrows into `GeminiProvider` / `StubProvider` from main (arrow stops at container border)

### Do NOT write on image

- `COLUMN 1`, `ROW 1`, `Container 1`, `#1`, `triad-rag`, `Stack`, `A1`
- Arrows to inner implementation classes

**Legend (tiny):** `—— calls` · `- - - extends`

---

## 4. Gemini master prompt (copy-paste)

```
Draw a clean technical architecture diagram for the Generation service (question + retrieved context → LLM answer).

Match the RETRIEVAL diagram style: main.py tall box on the LEFT; bordered PACKAGE CONTAINERS on the RIGHT; sparse labeled arrows.

TITLE: "Generation — main.py workflow" (no triad-rag)

LAYOUT:
- TOP HEADER (not a container): env.toml [generation] + nested provider tables, prompts.toml, settings box, small registry note (_PROVIDER_REGISTRY)
- LEFT: tall "main.py (API entry)" with routes:
  GET /health, GET /models, POST /models/select, POST /generate
  POST /generate body: question + context
- RIGHT: ONE bordered container "ai_providers/":
  INSIDE border only:
    LEFT: provider_factory — make_provider, provider_implemented, _PROVIDER_REGISTRY
    RIGHT: BaseAIProvider on top (list _system_instruction, _user_content, generate)
           dashed vertical line down
           implementation box: GeminiProvider (google.genai), StubProvider (local fake)
  Factory + base + derived ALL inside the same container border.

STYLE: flat boxes, sans-serif, landscape 16:9, off-white background.
Colors: main.py light blue; config header light orange; factory light indigo; base light teal; impl white; container border light gray.

CONFIG DETAIL (header boxes):
- env.toml: default_provider, temperature, per-provider api_key and model aliases
- prompts.toml: system_prompt (separate from env)
- settings: active provider, model_alias, model, api_key, system_prompt

ARROWS — exactly 4:
1. env.toml → settings: "loads"
2. prompts.toml → settings: "system prompt"
3. main.py → settings: "reads settings"
4. main.py → ai_providers/ container border: "ai_providers"

Inside ai_providers/: dashed base→derived only. NO creates arrows. NO arrows to GeminiProvider/StubProvider from main.

FORBIDDEN: triad-rag, COLUMN/ROW labels, Stack, A1, extra arrows, mega wrapper around all packages, arrows to inner impl classes.

Output one diagram image.
```

---

## 5. Repair prompt

```
Redraw Generation diagram. Fix:

1. ONE bordered ai_providers/ container — factory + BaseAIProvider + Gemini/Stub ALL inside the border
2. main.py → ai_providers/ container border (not to inner classes)
3. Header: env.toml + prompts.toml + settings
4. Exactly 4 arrows (loads, system prompt, reads settings, ai_providers)
5. Remove COLUMN 1, ROW 1, triad-rag, Stack, extra arrows
6. Dashed vertical base → implementations inside container only

Title: Generation — main.py workflow
```

---

## 6. Checklist

**Layout**
- [ ] Title `Generation — main.py workflow`
- [ ] Config header with env.toml + prompts.toml + settings
- [ ] One `ai_providers/` bordered container
- [ ] main.py tall on left

**Content**
- [ ] Routes: health, models, models/select, generate
- [ ] GeminiProvider + StubProvider in impl box
- [ ] POST /generate shows question + context inputs

**Arrows (4)**
- [ ] loads · system prompt · reads settings · ai_providers
- [ ] Arrows hit container border / settings only
- [ ] Dashed base→derived inside container

**Forbidden**
- [ ] No COLUMN/ROW/triad-rag/Stack/A1
