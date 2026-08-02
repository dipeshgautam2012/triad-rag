# triad-rag

`triad-rag` is a document-grounded RAG system built on three isolated HTTP microservices (**Retrieval**, **Orchestrator**, **Generation**) with zero shared Python code. It guarantees that answers are derived strictly from your ingested files—not model hallucinations.

---

## Architecture & System Overview

Each service operates independently over HTTP. Retrieval handles embeddings and search; Generation queries the LLM using retrieved passages; the Orchestrator coordinates the workflow.

![System overview](docs/diagrams/system-overview.png)

| Service | Endpoint | Role | Talks To |
|---|---|---|---|
| **Chat UI** (`ui/chat.py`) | Streamlit | User Q&A interface | Orchestrator (`:8100`) |
| **Index UI** (`retrieval/ui/index.py`) | Streamlit | File upload, collection management, search test | Retrieval (`:8101`) |
| **Eval UI** (`eval/ui/run.py`) | Streamlit | Golden-set metrics & evaluation dashboard | Orchestrator + Retrieval + Generation |
| **Orchestrator** | `:8100` | Coordinates chat queries and collection settings | Retrieval + Generation |
| **Retrieval** | `:8101` | Document ingestion, chunking, indexes (`corpus/` + `index_store/`) | Local storage / vector DB |
| **Generation** | `:8102` | LLM text generation strictly from passed context | External LLM Provider |

* **Ingest:** Index UI → Retrieval (saves files, chunks data, builds indexes).
* **Chat:** Chat UI → Orchestrator → Retrieval (fetches passages) → Generation (synthesizes answer).
* **Search Modes:** Configured **once per collection** at upload (`chroma`, `bm25`, or `hybrid`). Optional reranking and query expansion can be toggled per search. Retrieval never touches the LLM; Generation never searches files.

Detailed design specs: [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick Start

Prerequisites: **Python 3.10+** (project root: `triad-rag/`).

### 1. Setup Environment
```bash
cd triad-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.toml.bak env.toml
```
* **Configuration:** Edit `env.toml` to set your `api_key` under `[generation.google]` (or use `stub` for local testing without an API key), and verify `corpus_dir` / `index_store_dir` under `[retrieval]`.
* *Note:* The first upload will automatically download the default embedding model (`all-MiniLM-L6-v2`).

### 2. Run the Microservices
Open **three separate terminals**, activate the virtual environment in each, and launch the respective service:

```bash
# Terminal 1 — Orchestrator
cd orchestrator && python -m app.main

# Terminal 2 — Retrieval
cd retrieval && python -m app.main

# Terminal 3 — Generation
cd generation && python -m app.main
```
*(Optional flags: `--host 0.0.0.0`, `--no-reload`).*

Verify services are running:
```bash
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1:8101/health
curl -s http://127.0.0.1:8102/health
```

### 3. Launch the UIs
Run any of the Streamlit interfaces from the project root (with services active):

* **Index UI** (Upload files & test search): `streamlit run retrieval/ui/index.py`
* **Chat UI** (Interactive Q&A): `streamlit run ui/chat.py`
* **Eval UI** (Golden-set metrics): `streamlit run eval/ui/run.py`

### 4. First-Time Workflow
1. Open **Index UI**, upload a `.pdf` or `.txt` file, and assign an `index_id` (collection name) along with your preferred chunker and indexer.
2. Use the **Query** tab in the Index UI to verify passage retrieval.
3. Open **Chat UI**, select your collection, and start asking grounded questions.

---

## Chunker & Indexer Guide

Chunkers and indexers are selected **once per collection** upon initial upload. Subsequent uploads to the same `index_id` inherit these settings.

* **Supported Formats:** `.txt` and `.pdf` only. PDFs are parsed as plain text per page (no native table/layout extraction). Re-uploading a file overwrites its existing chunks.

### Chunkers
* **`simple` (Default):** Fixed-size splits. Ideal for general PDFs and text documents.
* **`markdown`:** Splits on `#` headings (requires Markdown `.txt` files; convert PDFs first).
* **`hierarchical`:** Creates small search hits that can return wider parent text contexts when expand is enabled (Use with `chroma` or `hybrid`; **do not use with `bm25` alone**).
* **`sentence_window`:** FAQ-style approach; searches an anchor sentence and returns a surrounding context window.
* **`semantic`:** Splits documents dynamically by topic change (requires embedding model during ingestion).

### Indexers
* **`hybrid` (Recommended):** Combines semantic vector search (`chroma`) and keyword search (`bm25`).
* **`chroma`:** Pure semantic embedding search.
* **`bm25`:** Pure keyword search (requires configuring `sparse_backend` as `json_bm25` or `sqlite_bm25` in `env.toml`).

For deep dives, see [`docs/chunking-strategies.md`](docs/chunking-strategies.md) and [`docs/retrieval-strategies.md`](docs/retrieval-strategies.md).

---

## Evaluation

Batch-validate pipeline accuracy using a golden dataset: fixed questions paired with reference answers and optional source constraints. Eval writes versioned CSV reports to `eval/datasets/<dataset_id>/reports/`.

### Dataset Format (`golden.jsonl`)
```json
{
  "question": "What is the refund policy?",
  "ground_truth": "Full refund within 30 days of purchase.",
  "expected_source": "handbook.pdf",
  "expected_page": 12
}
```
*(Ingest all referenced files into a collection matching your dataset folder name before running evaluations).*

### Running Evaluations
```bash
# Via UI (services must be active)
streamlit run eval/ui/run.py

# Via CLI
python eval/run_eval.py --dataset my_dataset
python eval/run_eval.py --dataset my_dataset --skip-faithfulness
python eval/run_eval.py --dataset my_dataset --index-id my_collection --top-k 5
```
*(Faithfulness scoring evaluates whether the LLM answer matches the retrieved text; use `--skip-faithfulness` if running with the `stub` LLM provider).*

---

## Library Examples (No HTTP Required)

To smoke-test the ingestion and query logic directly in Python without spinning up microservices, use the scripts in `examples/` (defaults to `examples/sample.txt`, writing indexes to `examples/data/`):

```bash
python examples/index_and_query.py
python examples/index_and_query.py --indexer bm25 --query "keyword search" --top-k 5
python examples/hybrid_index_and_query.py --chunker hierarchical --no-expand
```

---

## API Reference & Quick Commands

| Task | Command |
|------|---------|
| **List collections** | `curl -s http://127.0.0.1:8101/indices` |
| **Ingest options** | `curl -s http://127.0.0.1:8101/ingest/options` |
| **Upload file** | `curl -s -X POST http://127.0.0.1:8101/ingest -F file=@doc.pdf -F index_id=default -F indexer=hybrid -F embedding_model=all-MiniLM-L6-v2 -F chunker_name=simple` |
| **Search passages** | `curl -s -X POST http://127.0.0.1:8101/retrieve -H 'Content-Type: application/json' -d '{"query":"What is RAG?","top_k":3,"index_id":"default","rerank":false,"expand":true}'` |
| **Chat query** | `curl -s -X POST http://127.0.0.1:8100/query -H 'Content-Type: application/json' -d '{"question":"What is RAG?","index_id":"default","top_k":3,"rerank":false}'` |
| **List LLM models** | `curl -s http://127.0.0.1:8102/models` |

*Interactive Swagger API Docs:* `:8100/docs`, `:8101/docs`, `:8102/docs`.

---

## Configuration

Manage settings in `env.toml` (copied from `env.toml.bak`):
* **`[orchestrator]`**: Service URLs, timeouts, retry parameters.
* **`[retrieval]`**: Storage paths, default chunkers, embedding models, `sparse_backend`, expansion rules.
* **`[generation]`**: `default_provider`, temperature settings, and provider blocks (e.g., `[generation.google]`).
* **Prompts:** Customize system instructions in `generation/prompts.toml`.
* **Environment Overrides:** Supported via prefixes `ORCH_*`, `RET_*`, `GEN_*`.

---

## UI & Architecture Gallery

### Index UI & Chat UI
<p align="center">
  <img src="docs/diagrams/ui-index-screenshot.png" alt="Index UI" width="800" style="border: 1px solid #d0d7de; border-radius: 8px;" />
</p>
<p align="center">
  <img src="docs/diagrams/ui-chat-screenshot.png" alt="Chat UI" width="800" style="border: 1px solid #d0d7de; border-radius: 8px;" />
</p>

### Supplementary Documentation
* **Design Specs:** [`docs/DESIGN.md`](docs/DESIGN.md)
* **Chunking Strategies:** [`docs/chunking-strategies.md`](docs/chunking-strategies.md)
* **Retrieval Strategies:** [`docs/retrieval-strategies.md`](docs/retrieval-strategies.md)
* **Workflow Diagrams:** Orchestrator (`orchestrator_main.png`), Retrieval (`retrieval_main_workflow.png`), Generation (`generation_main_workflow.png`), Metadata (`retrieval_metadata_structure.png`), Technical Layout (`technical_architecture.png`).

---

## License

Licensed under the [MIT License](LICENSE).

---

## Citation

```bibtex
@software{gautam2026triadrag,
  author  = {Gautam, Dipesh},
  title   = {{triad-rag}: Document-grounded {RAG} with {HTTP} microservices},
  year    = {2026},
  url     = {https://github.com/dipeshgautam2012/triad-rag}
}
```
Plain text:
> Gautam, D. (2026). *triad-rag: Document-grounded RAG with HTTP microservices*. https://github.com/dipeshgautam2012/triad-rag