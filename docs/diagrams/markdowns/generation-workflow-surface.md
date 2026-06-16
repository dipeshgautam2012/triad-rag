# Generation service — API surface

Structural reference for diagram generation. Root: `generation/app/`

---

## config

### `app.config`

| Symbol | Kind | Notes |
|--------|------|-------|
| `settings` | instance | Active provider, model, API key, temperature, limits, `system_prompt` |
| `providers()` | function | `() -> dict` — catalog from `env.toml` nested `[generation.*]` tables |
| `list_models()` | function | `() -> dict` — active selection + implemented providers/models |
| `set_model(provider, model_alias)` | function | Runtime switch (in-memory) |

**Sources:**
- `env.toml` `[generation]` — `default_provider`, `temperature`, token limits
- `env.toml` `[generation.google]`, `[generation.stub]`, … — `api_key`, `default_model`, `[generation.*.models]` aliases
- `generation/prompts.toml` — `system_prompt` (not in env.toml)

**Called from:** `app.main` (`settings`, `list_models`, `set_model`, `providers`, `provider_implemented`)

---

## main (HTTP entry)

### `app.main`

| Route | Handler | Calls |
|-------|---------|-------|
| GET `/health` | `health()` | — |
| GET `/models` | `get_models()` | `list_models()` |
| POST `/models/select` | `select_model()` | `providers()`, `provider_implemented()`, `set_model()` |
| POST `/generate` | `generate()` | `make_provider(settings).generate(question, context)` |

**Request bodies:**
- `GenerateRequest` — `question`, `context`
- `ModelChoice` — `provider`, `model_alias`

**Response:** `GenerateResponse` — `answer`, `model`

---

## ai_providers

### `app.ai_providers.provider_factory`

| Symbol | Prototype |
|--------|-----------|
| `_PROVIDER_REGISTRY` | `google` → `GeminiProvider`, `stub` → `StubProvider` |
| `provider_implemented(name)` | `(name: str) -> bool` |
| `make_provider(cfg)` | `(cfg) -> BaseAIProvider` |

**Called from:** `app.main.generate`, `app.main.select_model`, `app.config._active_provider`, `list_models`

### `app.ai_providers.base_provider`

| Class | Methods |
|-------|---------|
| `BaseAIProvider` | `__init__(cfg)` · `_system_instruction()` · `_user_content(question, context)` · `generate(question, context, output_schema=None)` |

### `app.ai_providers.gemini_provider`

| Class | `generate` → Google `genai.Client.models.generate_content` |
|-------|-------------------------------------------------------------|

### `app.ai_providers.stub_provider`

| Class | `generate` → fake answer quoting question + context preview |

---

## Call summary

```
POST /generate
  → make_provider(settings)
  → BaseAIProvider.generate(question, context)
       → _system_instruction()  (from prompts.toml via settings)
       → _user_content(question, context)
       → GeminiProvider | StubProvider
```
