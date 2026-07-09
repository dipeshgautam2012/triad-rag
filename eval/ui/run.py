"""
Eval UI — run golden.jsonl against /retrieve + /query and write metrics.

Run from ``triad-rag``::

    streamlit run eval/ui/run.py

Golden sets live under ``eval/datasets/<dataset_id>/golden.jsonl``.
Each run writes ``eval/datasets/<dataset_id>/reports/<YYYYMMDDTHHMMSS>/report.csv``.
"""

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from run_eval import (  # noqa: E402
    DEFAULT_DATASET_ID,
    EvalConfig,
    golden_path,
    list_datasets,
    list_report_runs,
    load_golden_rows,
    load_golden_text,
    load_report_csv,
    missing_golden_sources,
    report_to_csv_text,
    run_eval,
    select_generation_model,
    summarize_metrics,
    write_report_run,
)

DEFAULT_RETRIEVAL = os.environ.get("RETRIEVAL_API_URL", "http://127.0.0.1:8101")
DEFAULT_ORCHESTRATOR = os.environ.get("ORCHESTRATOR_API_URL", "http://127.0.0.1:8100")
DEFAULT_GENERATION = os.environ.get("GENERATION_API_URL", "http://127.0.0.1:8102")


@st.cache_data(ttl=15)
def _fetch_ingest_options(api_base: str) -> tuple[dict[str, object], str | None]:
    try:
        r = httpx.get(f"{api_base.rstrip('/')}/ingest/options", timeout=httpx.Timeout(30.0))
    except httpx.RequestError as e:
        return ({}, str(e))
    if r.status_code >= 400:
        return ({}, f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data if isinstance(data, dict) else {}, None)


@st.cache_data(ttl=15)
def _fetch_models(orchestrator_base: str) -> tuple[dict, str | None]:
    try:
        r = httpx.get(f"{orchestrator_base.rstrip('/')}/models", timeout=httpx.Timeout(20.0))
    except httpx.RequestError as e:
        return ({}, str(e))
    if r.status_code >= 400:
        return ({}, f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data if isinstance(data, dict) else {}, None)


@st.cache_data(ttl=15)
def _fetch_indices(retrieval_base: str) -> tuple[list[str], str | None]:
    try:
        r = httpx.get(f"{retrieval_base.rstrip('/')}/indices", timeout=httpx.Timeout(30.0))
    except httpx.RequestError as e:
        return ([], str(e))
    if r.status_code >= 400:
        return ([], f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    rows_raw = data.get("files")
    if isinstance(rows_raw, list):
        ids = sorted({str(x.get("index_id", "")).strip() for x in rows_raw if isinstance(x, dict)})
        ids = [i for i in ids if i]
        return (ids, None)
    ids_raw = data.get("indices")
    if isinstance(ids_raw, list):
        ids = sorted({str(x).strip() for x in ids_raw if str(x).strip()})
        return (ids, None)
    return ([], "Invalid /indices response")


@st.cache_data(ttl=15)
def _fetch_corpus_files(retrieval_base: str, index_id: str) -> tuple[list[str], str | None]:
    from urllib.parse import quote

    try:
        r = httpx.get(
            f"{retrieval_base.rstrip('/')}/indices/{quote(index_id, safe='')}/files",
            timeout=httpx.Timeout(30.0),
        )
    except httpx.RequestError as e:
        return ([], str(e))
    if r.status_code >= 400:
        return ([], f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    raw = data.get("files") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return ([], None)
    return ([str(f) for f in raw if str(f).strip()], None)


def _health_label(base: str, path: str = "/health") -> str:
    try:
        r = httpx.get(f"{base.rstrip('/')}{path}", timeout=httpx.Timeout(5.0))
        if r.status_code == 200:
            return ":green-badge[ok]"
        return f":red-badge[HTTP {r.status_code}]"
    except httpx.RequestError:
        return ":red-badge[down]"


def _provider_section(orchestrator_url: str) -> tuple[str, str]:
    st.subheader(":material/smart_toy: Generation provider")
    models_data, models_err = _fetch_models(orchestrator_url)
    if models_err:
        st.warning(f"Could not load `GET /models`. ({models_err})")
        return ("", "")

    active_provider = str(models_data.get("provider") or "")
    active_alias = str(models_data.get("model_alias") or "")
    active_model = str(models_data.get("model") or "")
    st.markdown(f":violet-badge[:material/smart_toy: {active_provider} / {active_alias}]")
    st.caption(f"Resolves to `{active_model}` · applied before each eval run")

    providers_raw = models_data.get("providers") or {}
    providers = (
        {str(k): v for k, v in providers_raw.items() if isinstance(v, dict)}
        if isinstance(providers_raw, dict)
        else {}
    )
    provider_names = sorted(providers.keys())
    if not provider_names:
        st.warning("No providers configured on generation service.")
        return (active_provider, active_alias)

    c1, c2 = st.columns(2)
    with c1:
        prov_i = provider_names.index(active_provider) if active_provider in provider_names else 0
        selected_provider = st.selectbox("Provider", options=provider_names, index=prov_i)
    prov_entry = providers.get(selected_provider) or {}
    aliases_raw = prov_entry.get("models") or {}
    aliases = sorted(str(k) for k in aliases_raw.keys()) if isinstance(aliases_raw, dict) else []
    with c2:
        if not aliases:
            st.caption("No models for this provider.")
            return (selected_provider, active_alias)
        alias_i = aliases.index(active_alias) if active_alias in aliases else 0
        selected_alias = st.selectbox(
            "Model",
            options=aliases,
            index=alias_i,
            format_func=lambda a: f"{a} → {aliases_raw.get(a, '')}",
        )
    return (selected_provider, selected_alias)


def main() -> None:
    st.set_page_config(
        page_title="triad-rag · Eval",
        page_icon=":material/fact_check:",
        layout="wide",
    )
    st.title(":material/fact_check: RAG evaluation")
    st.markdown(
        "Run a dataset **golden.jsonl** against **retrieval** and **orchestrator**. "
        "Reports are versioned by timestamp under the same dataset folder."
    )
    st.divider()

    st.header(":material/cable: Connection")
    c1, c2 = st.columns(2)
    with c1:
        retrieval_url = st.text_input(
            "Retrieval URL",
            value=DEFAULT_RETRIEVAL,
            help="POST /retrieve — port 8101",
        ).strip().rstrip("/")
    with c2:
        orchestrator_url = st.text_input(
            "Orchestrator URL",
            value=DEFAULT_ORCHESTRATOR,
            help="POST /query and POST /models/select — port 8100",
        ).strip().rstrip("/")

    if not retrieval_url or not orchestrator_url:
        st.warning("Enter both service URLs.")
        st.stop()

    h1, h2 = st.columns(2)
    with h1:
        st.caption(f"Retrieval {_health_label(retrieval_url)}")
    with h2:
        st.caption(f"Orchestrator {_health_label(orchestrator_url)}")

    st.caption(
        f"Answers use **generation** at `{DEFAULT_GENERATION}` via orchestrator `/query`. "
        f"Pick provider below (orchestrator forwards to generation)."
    )

    st.divider()
    dataset_ids = list_datasets()
    if not dataset_ids:
        st.error(f"No datasets found under `{_EVAL_DIR / 'datasets'}/*/golden.jsonl`.")
        st.stop()

    st.header(":material/folder: Dataset")
    default_ds_i = dataset_ids.index(DEFAULT_DATASET_ID) if DEFAULT_DATASET_ID in dataset_ids else 0
    dataset_id = st.selectbox(
        "Dataset folder",
        options=dataset_ids,
        index=default_ds_i,
        help="Each folder contains golden.jsonl and a reports/ subfolder.",
    )
    golden_file = golden_path(dataset_id)
    st.caption(f"Golden file: `{golden_file.relative_to(_EVAL_DIR.parent)}`")

    st.divider()
    st.header(":material/tune: Run settings")

    ingest_opts, opts_err = _fetch_ingest_options(retrieval_url)
    if opts_err:
        st.caption(f"Could not load retrieval config ({opts_err}).")
    elif ingest_opts:
        with st.expander("Server config (`env.toml`)", icon=":material/tune:", expanded=False):
            st.caption("Eval uses `search_expand` and `rerank_enabled` from config (no per-run overrides).")
            st.json(ingest_opts)

    selected_provider, selected_alias = _provider_section(orchestrator_url)

    index_ids, idx_err = _fetch_indices(retrieval_url)
    if idx_err:
        st.warning(f"Could not load indices ({idx_err}).")
    if not index_ids:
        st.warning("No saved indexes on retrieval. Ingest documents in the Index UI first.")
        st.stop()

    default_index_i = index_ids.index(dataset_id) if dataset_id in index_ids else 0
    index_id = st.selectbox(
        "Index to evaluate",
        options=index_ids,
        index=default_index_i,
        help="Defaults to the dataset folder name when that index exists.",
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5)
    with s2:
        timeout_s = st.number_input("Timeout (seconds)", min_value=10.0, max_value=600.0, value=120.0)
    with s3:
        skip_faithfulness = st.checkbox(
            "Skip faithfulness judge",
            value=False,
            help="Skip the extra LLM grounding check after each answer.",
        )

    st.divider()
    st.header(":material/list_alt: Golden set")

    golden_source = st.radio(
        "Source",
        options=[f"Dataset: {dataset_id}/golden.jsonl", "Upload JSONL (one-off)"],
        horizontal=True,
    )

    golden_rows: list | None = None
    golden_error: str | None = None

    if golden_source.startswith("Dataset:"):
        try:
            golden_rows = load_golden_rows(golden_file)
            st.caption(f"{len(golden_rows)} question(s) loaded.")
        except ValueError as e:
            golden_error = str(e)
    else:
        uploaded = st.file_uploader("golden.jsonl", type=["jsonl", "txt", "json"])
        if uploaded is None:
            st.info("Upload a JSONL file (one JSON object per line).")
            st.stop()
        try:
            golden_rows = load_golden_text(
                uploaded.getvalue().decode("utf-8", errors="replace"),
                label=uploaded.name,
            )
            st.caption(f"{len(golden_rows)} question(s) from upload (not saved to dataset folder).")
        except ValueError as e:
            golden_error = str(e)

    if golden_error:
        st.error(golden_error)
        st.stop()
    if not golden_rows:
        st.warning("Golden set is empty.")
        st.stop()

    corpus_files, corpus_err = _fetch_corpus_files(retrieval_url, index_id)
    if corpus_err:
        st.warning(f"Could not verify corpus files ({corpus_err}).")
    else:
        missing = missing_golden_sources(golden_rows, set(corpus_files))
        if missing:
            st.error(
                "Golden `expected_source` files not in this index: "
                + ", ".join(f"`{name}`" for name in missing)
            )
            st.stop()

    with st.expander("Preview golden rows", expanded=False):
        st.dataframe(
            [
                {
                    "question": r.question,
                    "expected_source": r.expected_source,
                    "expected_page": "" if r.expected_page is None else str(r.expected_page),
                }
                for r in golden_rows
            ],
            use_container_width=True,
            hide_index=True,
        )

    past_runs = list_report_runs(dataset_id)
    run_labels = ["— new run only —"] + [p.name for p in past_runs]
    selected_run_label = st.selectbox(
        "Past report versions",
        options=run_labels,
        help="Load a previous report.csv from this dataset's reports/ folder.",
    )

    st.divider()
    run_clicked = st.button(":material/play_arrow: Run evaluation", type="primary")

    if run_clicked:
        config = EvalConfig(
            retrieval_url=retrieval_url,
            orchestrator_url=orchestrator_url,
            generation_url=DEFAULT_GENERATION,
            index_id=index_id,
            top_k=int(top_k),
            timeout_s=float(timeout_s),
            skip_faithfulness=skip_faithfulness,
        )
        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(current: int, total: int, question: str) -> None:
            progress.progress(current / total, text=f"[{current}/{total}] {question[:100]}")

        with st.spinner("Running eval…"):
            with httpx.Client(timeout=config.timeout_s) as client:
                if selected_provider and selected_alias:
                    try:
                        select_generation_model(
                            client,
                            orchestrator_url,
                            selected_provider,
                            selected_alias,
                        )
                    except Exception as e:
                        progress.empty()
                        st.error(f"Could not set provider/model: {e}")
                        st.stop()

            report_rows, hit_scores, overlap_scores, faith_scores = run_eval(
                golden_rows,
                config,
                on_progress=on_progress,
            )

        progress.empty()
        summary = summarize_metrics(hit_scores, overlap_scores, faith_scores, top_k=config.top_k)
        run_meta = {
            "dataset_id": dataset_id,
            "index_id": index_id,
            "provider": selected_provider,
            "model_alias": selected_alias,
            "top_k": config.top_k,
            "skip_faithfulness": skip_faithfulness,
            "metrics": summary,
        }
        if golden_source.startswith("Upload"):
            run_meta["golden_source"] = "upload"

        out_path = write_report_run(dataset_id, report_rows, run_meta)
        _fetch_models.clear()

        st.session_state["eval_report"] = report_rows
        st.session_state["eval_summary"] = summary
        st.session_state["eval_skip_faithfulness"] = skip_faithfulness
        st.session_state["eval_report_path"] = str(out_path)
        status.success(f"Done — wrote `{out_path.relative_to(_EVAL_DIR.parent)}`")

    if selected_run_label != "— new run only —":
        run_dir = golden_file.parent / "reports" / selected_run_label
        report_path = run_dir / "report.csv"
        summary_path = run_dir / "summary.json"
        if report_path.is_file():
            report_rows = load_report_csv(report_path)
            summary = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    summary = {}
            metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
            st.session_state["eval_report"] = report_rows
            st.session_state["eval_summary"] = metrics or summarize_metrics(
                [float(r.get("hit_at_k") or 0) for r in report_rows],
                [float(r.get("keyword_overlap") or 0) for r in report_rows],
                [
                    float(r.get("faithful") or 0)
                    for r in report_rows
                    if str(r.get("faithful", "")).strip() != ""
                ],
                top_k=int(metrics.get("top_k") or 5),
            )
            st.session_state["eval_skip_faithfulness"] = bool(summary.get("skip_faithfulness"))
            st.session_state["eval_report_path"] = str(report_path)
            if not run_clicked:
                st.info(f"Viewing report **`{selected_run_label}`**")

    if "eval_report" not in st.session_state:
        st.stop()

    summary = st.session_state["eval_summary"]
    report_rows = st.session_state["eval_report"]
    report_path_s = st.session_state.get("eval_report_path", "")

    st.header(":material/analytics: Summary")
    if report_path_s:
        st.caption(f"Report: `{report_path_s}`")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions", int(summary.get("questions", len(report_rows))))
    top_k_val = int(summary.get("top_k", 5))
    m2.metric(f"hit@{top_k_val}", f"{float(summary.get('hit_at_k', 0)):.1%}")
    m3.metric("Keyword overlap", f"{float(summary.get('keyword_overlap', 0)):.1%}")
    if st.session_state.get("eval_skip_faithfulness"):
        m4.metric("Faithfulness", "skipped")
    else:
        m4.metric("Faithfulness", f"{float(summary.get('faithfulness', 0)):.1%}")

    st.download_button(
        "Download report.csv",
        data=report_to_csv_text(report_rows),
        file_name=Path(report_path_s).name if report_path_s else "report.csv",
        mime="text/csv",
    )

    st.header(":material/table: Per-question results")
    st.dataframe(report_rows, use_container_width=True, hide_index=True)

    errors = [r for r in report_rows if r.get("error")]
    if errors:
        st.warning(f"{len(errors)} row(s) had errors — see the **error** column.")


if __name__ == "__main__":
    main()
