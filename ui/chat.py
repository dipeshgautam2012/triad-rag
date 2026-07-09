"""
Chat UI — ask questions via the orchestrator (retrieval + generation).

Run from ``triad-rag``::

    streamlit run ui/chat.py

Optional: ``export ORCHESTRATOR_API_URL=http://127.0.0.1:8100``
"""

import os

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("ORCHESTRATOR_API_URL", "http://127.0.0.1:8100")
DEFAULT_RETRIEVAL = os.environ.get("RETRIEVAL_API_URL", "http://127.0.0.1:8101")


def _opt_bool(opts: dict[str, object], key: str, default: bool) -> bool:
    val = opts.get(key)
    return val if isinstance(val, bool) else default


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
def _fetch_models(api_base: str) -> tuple[dict, str | None]:
    try:
        r = httpx.get(f"{api_base.rstrip('/')}/models", timeout=httpx.Timeout(20.0))
    except httpx.RequestError as e:
        return ({}, str(e))
    if r.status_code >= 400:
        return ({}, f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data if isinstance(data, dict) else {}, None)


@st.cache_data(ttl=15)
def _fetch_indices(api_base: str) -> tuple[list[dict[str, str]], str | None]:
    """Returns (index rows, error_message)."""
    try:
        r = httpx.get(f"{api_base.rstrip('/')}/indices", timeout=httpx.Timeout(10.0))
    except httpx.RequestError as e:
        return ([], str(e))
    if r.status_code >= 400:
        return ([], f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    rows_raw = data.get("files")
    if isinstance(rows_raw, list):
        rows: list[dict[str, str]] = []
        for x in rows_raw:
            if not isinstance(x, dict):
                continue
            index_id = str(x.get("index_id", "")).strip()
            if not index_id:
                continue
            rows.append(
                {
                    "index_id": index_id,
                    "description": str(x.get("description", "")).strip(),
                }
            )
        return (sorted(rows, key=lambda x: x["index_id"]), None)
    ids = data.get("indices")
    if not isinstance(ids, list):
        return ([], "Invalid /indices response")
    rows = [{"index_id": str(x), "description": ""} for x in sorted({str(x).strip() for x in ids}) if str(x).strip()]
    return (rows, None)


def _sidebar_model_section(base: str) -> None:
    st.subheader(":material/smart_toy: LLM model")
    models_data, models_err = _fetch_models(base)
    if models_err:
        st.warning(f"Could not load `GET /models`. ({models_err})")
        return

    active_provider = str(models_data.get("provider") or "")
    active_alias = str(models_data.get("model_alias") or "")
    active_model = str(models_data.get("model") or "")
    st.markdown(f":violet-badge[:material/smart_toy: {active_provider} / {active_alias}]")
    st.caption(f"Resolves to `{active_model}`")

    providers_raw = models_data.get("providers") or {}
    providers = (
        {str(k): v for k, v in providers_raw.items() if isinstance(v, dict)}
        if isinstance(providers_raw, dict)
        else {}
    )
    provider_names = sorted(providers.keys())
    if not provider_names:
        return

    with st.expander("Change model", icon=":material/swap_horiz:", expanded=False):
        prov_default_i = (
            provider_names.index(active_provider) if active_provider in provider_names else 0
        )
        selected_provider = st.selectbox(
            "Provider",
            options=provider_names,
            index=prov_default_i,
            key="llm_provider",
        )
        prov_entry = providers.get(selected_provider) or {}
        aliases_raw = prov_entry.get("models") or {}
        aliases = (
            sorted(str(k) for k in aliases_raw.keys())
            if isinstance(aliases_raw, dict)
            else []
        )
        if not aliases:
            st.caption("No models listed for this provider.")
            return
        alias_default_i = aliases.index(active_alias) if active_alias in aliases else 0
        selected_alias = st.selectbox(
            "Model",
            options=aliases,
            index=alias_default_i,
            format_func=lambda a: f"{a} → {aliases_raw.get(a, '')}",
            key="llm_alias",
        )
        unchanged = selected_provider == active_provider and selected_alias == active_alias
        if st.button(
            "Apply model",
            icon=":material/check:",
            key="btn_apply_model",
            disabled=unchanged,
            help="Pick a different provider/model to enable." if unchanged else None,
            use_container_width=True,
        ):
            try:
                r = httpx.post(
                    f"{base}/models/select",
                    json={"provider": selected_provider, "model_alias": selected_alias},
                    timeout=httpx.Timeout(30.0),
                )
            except httpx.RequestError as e:
                st.error(f"Request failed: {e}")
            else:
                if r.status_code >= 400:
                    st.error(f"HTTP {r.status_code}: {r.text}")
                else:
                    _fetch_models.clear()
                    st.success("Model updated.", icon=":material/check_circle:")
                    st.rerun()


def _sidebar_index_section(base: str) -> str:
    st.subheader(":material/folder_special: Index")
    listed, idx_err = _fetch_indices(base)
    if idx_err:
        st.warning(f"Could not load `GET /indices` — list may be incomplete. ({idx_err})")

    if not listed:
        st.info("No saved indexes yet. Create one in the **Index UI** (`retrieval/ui/index.py`) first.")
        if st.button("Refresh list", icon=":material/refresh:", use_container_width=True):
            _fetch_indices.clear()
            _fetch_models.clear()
            _fetch_ingest_options.clear()
            st.rerun()
        return ""

    descriptions = {x["index_id"]: x["description"] for x in listed}
    index_ids = [x["index_id"] for x in listed]
    default_i = index_ids.index("default") if "default" in index_ids else 0
    index_id = st.selectbox(
        "Saved index",
        options=index_ids,
        index=default_i,
        help="Indices returned by the orchestrator from `GET /indices` (snapshot on disk).",
    )
    snapshot_desc = (descriptions.get(index_id) or "").strip()
    if snapshot_desc:
        st.info(snapshot_desc, icon=":material/notes:")

    if st.button("Refresh list", icon=":material/refresh:", use_container_width=True):
        _fetch_indices.clear()
        _fetch_models.clear()
        _fetch_ingest_options.clear()
        st.rerun()
    return index_id


def main() -> None:
    st.set_page_config(
        page_title="triad-rag · Chat",
        page_icon=":material/forum:",
        layout="centered",
    )

    with st.sidebar:
        st.header(":material/settings: Settings")

        with st.expander("Connection", icon=":material/cable:", expanded=False):
            base = st.text_input(
                "Orchestrator base URL",
                value=DEFAULT_API,
                help="Base URL only (no trailing path). Example: http://127.0.0.1:8100",
            ).strip().rstrip("/")
            retrieval_base = st.text_input(
                "Retrieval base URL (config display)",
                value=DEFAULT_RETRIEVAL,
                help="Used only to show `env.toml` retrieval settings.",
            ).strip().rstrip("/")
        if not base:
            st.warning("Enter the orchestrator API URL to continue.")
            st.stop()

        ingest_opts: dict[str, object] = {}
        if retrieval_base:
            ingest_opts, opts_err = _fetch_ingest_options(retrieval_base)
            if opts_err:
                st.caption(f"Retrieval config unavailable ({opts_err}).")
            elif ingest_opts:
                with st.expander("Server config (`env.toml`)", icon=":material/tune:", expanded=False):
                    st.caption("Expand uses `search_expand`. Rerank checkbox overrides `rerank_enabled`.")
                    st.json(ingest_opts)

        st.divider()
        _sidebar_model_section(base)

        st.divider()
        index_id = _sidebar_index_section(base)

        st.divider()
        with st.expander("Diagnostics", icon=":material/monitor_heart:", expanded=False):
            if st.button("Run GET /health", icon=":material/ecg:", key="orch_health"):
                try:
                    r = httpx.get(f"{base}/health", timeout=httpx.Timeout(20.0))
                except httpx.RequestError as e:
                    st.error(f"Request failed: {e}")
                else:
                    if r.status_code >= 400:
                        st.error(f"HTTP {r.status_code}: {r.text}")
                    else:
                        st.json(r.json())

    st.title(":material/forum: Chat with your documents")
    if not index_id:
        st.markdown("Pick an index in the sidebar after you ingest documents in the Index UI.")
        st.stop()

    st.markdown(
        "Ask questions in natural language. The orchestrator retrieves relevant chunks "
        f"from index `{index_id}` and the model answers using that context."
    )

    question = st.text_area(
        "Your question",
        placeholder="What is the refund policy?",
        height=120,
        label_visibility="collapsed",
    )
    col_k, col_rr = st.columns([3, 1], vertical_alignment="bottom")
    with col_k:
        top_k = st.slider(
            "Chunks to retrieve (top_k)",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of retrieval hits passed into the model as context.",
        )
    with col_rr:
        use_rerank = st.checkbox(
            "Rerank results",
            value=_opt_bool(ingest_opts, "rerank_enabled", False),
            help="Re-score retrieved chunks with a cross-encoder before generation.",
            key="rerank_checkbox",
        )

    if st.button(
        "Ask",
        icon=":material/send:",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    ):
        body = {
            "question": question.strip(),
            "index_id": index_id,
            "top_k": int(top_k),
            "rerank": use_rerank,
        }
        try:
            with st.spinner("Calling orchestrator `POST /query`…"):
                r = httpx.post(
                    f"{base}/query",
                    json=body,
                    timeout=httpx.Timeout(180.0),
                )
        except httpx.RequestError as e:
            st.error(f"Request failed: {e}")
            return

        if r.status_code >= 400:
            st.error(f"HTTP {r.status_code}: {r.text}")
            return

        data = r.json()
        answer = str(data.get("answer", "")).strip()
        sources = data.get("sources") or []

        st.divider()
        st.header(":material/chat: Answer")
        st.markdown(answer or "_Empty answer._")
        st.caption(f"{len(sources)} source chunk(s) returned.")

        if sources:
            st.subheader(":material/library_books: Sources")
            for i, src in enumerate(sources):
                score = src.get("score")
                cid = src.get("chunk_id", "")
                meta = src.get("metadata") or {}
                title = (
                    f"[{i + 1}] {cid} — score {float(score):.4f}"
                    if isinstance(score, (int, float))
                    else f"[{i + 1}] {cid}"
                )
                with st.expander(title, icon=":material/article:", expanded=(i == 0)):
                    if meta:
                        st.caption(f"metadata: {meta}")
                    st.write(src.get("text", ""))


main()
