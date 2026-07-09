"""
Index UI — ingest documents and test retrieval by API URL and ``index_id``.

``index_id`` = logical corpus (``default`` = files in corpus root; else ``corpus/{index_id}/``).

Run from ``triad-rag``::

    streamlit run retrieval/ui/index.py

Or from ``triad-rag/retrieval``::

    streamlit run ui/index.py

Optional: ``export RETRIEVAL_API_URL=http://127.0.0.1:8101``
"""

import os
from urllib.parse import quote

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("RETRIEVAL_API_URL", "http://127.0.0.1:8101")

# Colored badges for the indexer recorded in index metadata.
_INDEXER_BADGES = {
    "chroma": ":blue-badge[:material/scatter_plot: chroma]",
    "vector": ":blue-badge[:material/scatter_plot: chroma]",  # legacy metadata
    "bm25": ":orange-badge[:material/manage_search: bm25]",
    "hybrid": ":violet-badge[:material/hub: hybrid]",
}


def _indexer_badge(name: str) -> str:
    return _INDEXER_BADGES.get(name, f":gray-badge[{name}]")


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


def _render_server_config(opts: dict[str, object]) -> None:
    if not opts:
        return
    with st.expander("Server config (`env.toml`)", icon=":material/tune:", expanded=False):
        st.caption(
            "Edit `[retrieval]` in `env.toml` and restart retrieval to change these. "
            "Chunk tuning applies on ingest; search defaults apply when a request omits expand/rerank."
        )
        st.json(opts)


@st.cache_data(ttl=15)
def _fetch_indices(api_base: str) -> tuple[list[dict[str, object]], str | None, str | None]:
    """Returns (index rows, error_message, index_store_dir from API)."""
    try:
        r = httpx.get(f"{api_base.rstrip('/')}/indices", timeout=httpx.Timeout(60.0))
    except httpx.RequestError as e:
        return ([], str(e), None)
    if r.status_code >= 400:
        return ([], f"HTTP {r.status_code}: {r.text[:200]}", None)

    data = r.json()
    store_dir = data.get("index_store_dir")
    store_s = str(store_dir).strip() if store_dir is not None else None
    rows_raw = data.get("files")
    rows: list[dict[str, object]] = []
    if isinstance(rows_raw, list):
        for x in rows_raw:
            if not isinstance(x, dict):
                continue
            rows.append(
                {
                    "index_id": str(x.get("index_id", "")),
                    "description": str(x.get("description", "")),
                    "embedding_model": str(x.get("embedding_model", "")),
                    "chunker": str(x.get("chunker", "")),
                    "indexer": str(x.get("indexer", "")),
                    "chunks": int(x.get("chunks", 0) or 0),
                }
            )
    return (sorted(rows, key=lambda r: str(r["index_id"])), None, store_s)


@st.cache_data(ttl=15)
def _fetch_corpus_files(api_base: str, index_id: str) -> tuple[list[str], str | None]:
    try:
        r = httpx.get(
            f"{api_base.rstrip('/')}/indices/{quote(index_id, safe='')}/files",
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


def main() -> None:
    st.set_page_config(
        page_title="triad-rag · Index builder",
        page_icon=":material/database:",
        layout="centered",
    )
    st.title(":material/database: Index builder")
    st.markdown(
        "Create and refresh search indexes for RAG: **Ingest** uploads into the corpus and rebuilds the index; "
        "**Query** runs `POST /retrieve`. Index **`default`** uses the corpus root; others use **`corpus/{index_id}/`**."
    )
    st.divider()

    st.header(":material/cable: Connection")
    base = st.text_input(
        "Retrieval base URL",
        value=DEFAULT_API,
        help="Base URL only (no trailing path). Example: http://127.0.0.1:8101",
    ).strip().rstrip("/")
    if not base:
        st.warning("Enter the retrieval API URL to continue.")
        st.stop()

    st.divider()
    st.header(":material/folder_special: Index")
    listed, idx_err, index_store_dir = _fetch_indices(base)
    ingest_opts, opts_err = _fetch_ingest_options(base)
    if idx_err:
        st.warning(f"Could not load `GET /indices` — list may be incomplete. ({idx_err})")
    if opts_err:
        st.warning(f"Could not load `GET /ingest/options`. ({opts_err})")
    else:
        _render_server_config(ingest_opts)

    st.divider()
    if not listed:
        st.info(
            "No saved indexes yet. Enable **Use a custom index_id** below to ingest your first "
            "(e.g. `default` for the corpus root)."
        )
    use_custom = st.checkbox(
        "Use a custom index_id (not only from the list)",
        value=not bool(listed),
        help="For ids that are not in GET /indices yet (e.g. before first ingest). "
        "Rules: letters, digits, underscore, hyphen, 1–64 chars.",
    )
    if use_custom:
        index_id = st.text_input(
            "Custom index_id",
            value="default",
            help="`default` = corpus root; otherwise files live under `corpus/{index_id}/`.",
            key="index_id_input",
        ).strip() or "default"
    elif not listed:
        index_id = ""
    else:
        c_sel, c_ref = st.columns([3, 1], vertical_alignment="bottom")
        with c_sel:
            index_ids = [str(r["index_id"]) for r in listed]
            default_i = index_ids.index("default") if "default" in index_ids else 0
            index_id = st.selectbox(
                "Saved index",
                options=index_ids,
                index=default_i,
                help="From GET /indices — Chroma collections that exist on disk.",
            )
        with c_ref:
            if st.button(
                "Refresh list",
                icon=":material/refresh:",
                use_container_width=True,
                help="Refetch GET /indices",
            ):
                _fetch_indices.clear()
                _fetch_ingest_options.clear()
                st.rerun()

    if not index_id:
        st.stop()

    rows_by_id = {str(r["index_id"]): r for r in listed}
    existing_ids = set(rows_by_id.keys())
    if use_custom and index_id in existing_ids:
        st.warning(
            f"**`{index_id}` already exists** (listed by `GET /indices`). "
            "The next ingest will **add to** that index (same Chroma collection), not create a separate one. "
            "To reuse the name for a **fresh** index, delete it in **Delete saved index** first, or choose another id."
        )
    elif use_custom and idx_err:
        st.caption("Index list failed to load above — could not verify whether this id already exists server-side.")

    snap = rows_by_id.get(index_id)

    st.subheader(":material/info: Index metadata")
    if index_store_dir:
        st.caption(f"**Index store directory:** `{index_store_dir}`")

    if snap:
        idxr = str(snap.get("indexer") or "").strip() or "chroma"
        emb = str(snap.get("embedding_model") or "").strip()
        chunker = str(snap.get("chunker") or "").strip()
        chunks_raw = snap.get("chunks", 0)
        chunk_count = chunks_raw if isinstance(chunks_raw, int) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", chunk_count)
        c2.metric("Indexer", idxr)
        c3.metric("Chunker", chunker or "—")
        badges = [_indexer_badge(idxr)]
        if chunker:
            badges.append(f":gray-badge[:material/content_cut: {chunker}]")
        if idxr != "bm25" and emb:
            badges.append(f":gray-badge[:material/memory: {emb}]")
        st.markdown(" ".join(badges))
        corpus_loc = "corpus root" if index_id == "default" else f"corpus/{index_id}/"
        st.caption(f"**Corpus:** `{corpus_loc}`")
    elif index_id:
        st.session_state.setdefault("ingest_desc_new", "")
        if st.session_state.get("ingest_custom_id_for_desc") != index_id:
            st.session_state["ingest_custom_id_for_desc"] = index_id
            st.session_state["ingest_desc_new"] = ""
        st.text_area(
            "Description for the new index (optional)",
            height=100,
            key="ingest_desc_new",
            help="Stored on the Chroma collection at first ingest. Max 500 chars.",
        )
    else:
        st.caption("Select a saved index above.")

    if snap:
        st.subheader(":material/notes: Index description")
        desc_saved = str(snap.get("description") or "").strip()

        if st.session_state.get("ingest_listed_index_tracker") != index_id:
            st.session_state["ingest_listed_index_tracker"] = index_id
            st.session_state["listed_desc_editing"] = False

        if desc_saved:
            st.info(desc_saved)
        else:
            st.caption("No description stored for this index yet.")

        if not st.session_state.get("listed_desc_editing"):
            if st.button("Update description", icon=":material/edit:", key="btn_listed_update_desc"):
                st.session_state["listed_desc_editing"] = True
                st.session_state["listed_desc_draft"] = desc_saved
                st.rerun()
        else:
            st.text_area(
                "Edit description",
                height=110,
                key="listed_desc_draft",
                help="Up to 500 characters. Saved via POST /indices/{id}/description.",
            )
            c_save, c_cancel = st.columns(2)
            with c_save:
                if st.button("Save description", icon=":material/save:", key="btn_listed_save_desc"):
                    try:
                        r = httpx.post(
                            f"{base.rstrip('/')}/indices/{index_id}/description",
                            json={"description": st.session_state.get("listed_desc_draft", "")},
                            timeout=httpx.Timeout(60.0),
                        )
                    except httpx.RequestError as e:
                        st.error(f"Request failed: {e}")
                    else:
                        if r.status_code >= 400:
                            st.error(f"HTTP {r.status_code}: {r.text}")
                        else:
                            st.session_state["listed_desc_editing"] = False
                            _fetch_indices.clear()
                            st.success("Description saved.", icon=":material/check_circle:")
                            st.json(r.json())
            with c_cancel:
                if st.button("Cancel", icon=":material/close:", key="btn_listed_cancel_desc"):
                    st.session_state["listed_desc_editing"] = False
                    st.rerun()

    st.divider()
    st.subheader(":material/monitor_heart: Service check")
    if st.button("GET /health", icon=":material/ecg:", key="btn_health"):
        try:
            r = httpx.get(f"{base}/health", timeout=httpx.Timeout(30.0))
        except httpx.RequestError as e:
            st.error(str(e))
        else:
            if r.status_code >= 400:
                st.error(f"HTTP {r.status_code}: {r.text}")
            else:
                st.json(r.json())

    st.divider()
    with st.expander("Delete saved index", icon=":material/delete_forever:", expanded=False):
        st.markdown(
            "Calls **DELETE /indices/{index_id}** — drops the Chroma collection and "
            "evicts the index from memory. **Corpus files are not deleted.**"
        )
        del_confirm = st.checkbox(
            f"I understand — delete index `{index_id}`",
            value=False,
            key="delete_index_confirm",
        )
        if st.button(
            "Delete index",
            icon=":material/delete:",
            disabled=not del_confirm,
            key="btn_delete_index",
            type="secondary",
        ):
            try:
                r = httpx.delete(
                    f"{base.rstrip('/')}/indices/{index_id}",
                    timeout=httpx.Timeout(30.0),
                )
            except httpx.RequestError as e:
                st.error(f"Request failed: {e}")
            else:
                if r.status_code >= 400:
                    st.error(f"HTTP {r.status_code}: {r.text}")
                else:
                    _fetch_indices.clear()
                    _fetch_ingest_options.clear()
                    st.success("Index deleted.", icon=":material/check_circle:")
                    st.json(r.json())

    st.divider()
    tab_ingest, tab_query = st.tabs(
        [":material/cloud_upload: Ingest", ":material/manage_search: Query"]
    )

    with tab_ingest:
        st.header(":material/cloud_upload: Upload & reindex")
        st.caption(
            "Upload a `.txt` or `.pdf` into the corpus for this index_id, "
            "then rebuild its index with the configuration below."
        )

        is_new_index = snap is None
        embedding_models = ingest_opts.get("embedding_models") or []
        if not isinstance(embedding_models, list):
            embedding_models = []
        embedding_models = [str(m) for m in embedding_models if str(m).strip()]
        default_embedding = str(ingest_opts.get("default_embedding_model") or "").strip()
        chunkers = ingest_opts.get("chunkers") or []
        if not isinstance(chunkers, list):
            chunkers = []
        chunkers = [str(c) for c in chunkers if str(c).strip()]
        default_chunker = str(ingest_opts.get("default_chunker") or "").strip()
        available_indexers = ingest_opts.get("indexers") or ["chroma"]
        if not isinstance(available_indexers, list):
            available_indexers = ["chroma"]
        available_indexers = [str(i) for i in available_indexers if str(i).strip()]
        default_indexer = str(ingest_opts.get("default_indexer") or "chroma").strip()

        if is_new_index:
            st.markdown(":material/tune: **Index configuration** — set at first ingest, locked afterward.")
            col_idx, col_chk = st.columns(2)
            with col_idx:
                idx_default_i = (
                    available_indexers.index(default_indexer)
                    if default_indexer in available_indexers
                    else 0
                )
                selected_indexer = st.selectbox(
                    "Indexer",
                    options=available_indexers or ["chroma"],
                    index=idx_default_i if available_indexers else 0,
                    disabled=not available_indexers,
                    help="chroma = dense embeddings · bm25 = keyword · hybrid = both. "
                    "Queries automatically use this choice (read from index metadata).",
                )
            with col_chk:
                chunk_default_i = (
                    chunkers.index(default_chunker) if default_chunker in chunkers else 0
                )
                selected_chunker = st.selectbox(
                    "Chunker",
                    options=chunkers or [default_chunker or "(none configured)"],
                    index=chunk_default_i if chunkers else 0,
                    disabled=not chunkers,
                    help="How documents are split into chunks before indexing.",
                )
                if selected_chunker == "markdown":
                    st.caption(
                        ":material/info: **Section-based (markdown)** — markdown text only. "
                        "Use a `.txt` file with `#` headings, or convert PDF/other formats to "
                        "markdown before upload. Raw PDFs are not section-split."
                    )
            # bm25 is keyword-based and computes no embeddings — the embedding model
            # only matters there when the semantic chunker uses it to find boundaries.
            embedding_needed = selected_indexer != "bm25" or selected_chunker == "semantic"
            if embedding_needed:
                emb_default_i = (
                    embedding_models.index(default_embedding)
                    if default_embedding in embedding_models
                    else 0
                )
                selected_embedding = st.selectbox(
                    "Embedding model",
                    options=embedding_models or [default_embedding or "(none configured)"],
                    index=emb_default_i if embedding_models else 0,
                    disabled=not embedding_models,
                    help="Used to embed chunks for chroma/hybrid search"
                    + (
                        "; with the semantic chunker it also determines chunk boundaries."
                        if selected_chunker == "semantic"
                        else "."
                    ),
                )
            else:
                selected_embedding = ""
                st.caption("No embedding model needed — `bm25` indexes keywords, not embeddings.")
        else:
            locked_indexer = str(snap.get("indexer") or "chroma").strip() or "chroma"
            locked_badges = [_indexer_badge(locked_indexer)]
            locked_badges.append(f":gray-badge[:material/content_cut: {snap.get('chunker', '')}]")
            if locked_indexer != "bm25":
                locked_badges.append(
                    f":gray-badge[:material/memory: {snap.get('embedding_model', '')}]"
                )
            st.markdown(" ".join(locked_badges) + " :small[:gray[(locked for this index)]]")
            selected_indexer = ""
            selected_embedding = ""
            selected_chunker = ""

        uploaded = st.file_uploader(
            "Document",
            type=["txt", "pdf"],
            help="Drag and drop or browse. Only .txt and .pdf.",
            key="ingest_file",
        )

        with st.expander("Corpus files", icon=":material/folder_open:", expanded=False):
            corpus_files, cf_err = _fetch_corpus_files(base, index_id)
            if cf_err:
                st.warning(f"Could not load corpus files. ({cf_err})")
            elif not corpus_files:
                st.caption("No `.txt` or `.pdf` files in this index corpus yet.")
            else:
                st.caption(f"**{len(corpus_files)}** file(s) — from `GET /indices/{{id}}/files`")
                file_to_delete = st.selectbox(
                    "File",
                    options=corpus_files,
                    key="corpus_file_select",
                )
                del_file_confirm = st.checkbox(
                    f"Delete `{file_to_delete}` from disk and remove its chunks from the index",
                    value=False,
                    key="delete_corpus_file_confirm",
                )
                if st.button(
                    "Delete file",
                    icon=":material/delete:",
                    disabled=not del_file_confirm,
                    key="btn_delete_corpus_file",
                    type="secondary",
                ):
                    try:
                        r = httpx.delete(
                            f"{base.rstrip('/')}/indices/{quote(index_id, safe='')}/files/{quote(file_to_delete, safe='')}",
                            timeout=httpx.Timeout(120.0),
                        )
                    except httpx.RequestError as e:
                        st.error(f"Request failed: {e}")
                    else:
                        if r.status_code >= 400:
                            st.error(f"HTTP {r.status_code}: {r.text}")
                        else:
                            _fetch_corpus_files.clear()
                            _fetch_indices.clear()
                            st.success("File deleted.", icon=":material/check_circle:")
                            st.json(r.json())

                st.divider()
                folder_note = (
                    f" and remove folder `corpus/{index_id}/`"
                    if index_id != "default"
                    else " from the corpus root (`default`)"
                )
                clear_confirm = st.checkbox(
                    f"Delete all corpus files{folder_note} — vector index is **not** deleted",
                    value=False,
                    key="clear_corpus_confirm",
                )
                if st.button(
                    "Delete all corpus files",
                    icon=":material/delete_sweep:",
                    disabled=not clear_confirm,
                    key="btn_clear_corpus",
                    type="secondary",
                ):
                    try:
                        r = httpx.delete(
                            f"{base.rstrip('/')}/indices/{quote(index_id, safe='')}/corpus",
                            timeout=httpx.Timeout(300.0),
                        )
                    except httpx.RequestError as e:
                        st.error(f"Request failed: {e}")
                    else:
                        if r.status_code >= 400:
                            st.error(f"HTTP {r.status_code}: {r.text}")
                        else:
                            _fetch_corpus_files.clear()
                            _fetch_indices.clear()
                            st.success("Corpus cleared.", icon=":material/check_circle:")
                            st.json(r.json())

        if st.button(
            "Upload and reindex",
            icon=":material/rocket_launch:",
            type="primary",
            disabled=uploaded is None,
            use_container_width=True,
            key="btn_ingest",
        ):
            if uploaded is None:
                st.error("Choose a file first.")
            else:
                suffix = (uploaded.name or "").lower().split(".")[-1]
                if suffix not in ("txt", "pdf"):
                    st.error("Only .txt and .pdf are allowed.")
                else:
                    files = {
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    }
                    if use_custom:
                        data = {
                            "index_id": index_id,
                            "index_description": st.session_state.get("ingest_desc_new", ""),
                        }
                    else:
                        existing_desc = str((rows_by_id.get(index_id) or {}).get("description") or "")
                        data = {"index_id": index_id, "index_description": existing_desc}
                    if is_new_index and selected_embedding:
                        data["embedding_model"] = selected_embedding
                    if is_new_index and selected_chunker:
                        data["chunker_name"] = selected_chunker
                    if is_new_index and selected_indexer:
                        data["indexer"] = selected_indexer
                    try:
                        with st.spinner("POST /ingest — uploading and rebuilding index…"):
                            r = httpx.post(
                                f"{base}/ingest",
                                files=files,
                                data=data,
                                timeout=httpx.Timeout(600.0),
                            )
                    except httpx.RequestError as e:
                        st.error(f"Request failed: {e}")
                    else:
                        if r.status_code >= 400:
                            st.error(f"HTTP {r.status_code}: {r.text}")
                        else:
                            _fetch_indices.clear()
                            _fetch_ingest_options.clear()
                            _fetch_corpus_files.clear()
                            st.success("Ingest complete.", icon=":material/check_circle:")
                            st.json(r.json())

    with tab_query:
        st.header(":material/manage_search: Search")
        recorded_indexer = str((snap or {}).get("indexer") or "chroma").strip()
        st.markdown(
            f"Searches index `{index_id}` with its recorded indexer "
            f"{_indexer_badge(recorded_indexer)} — embedding model and chunker are "
            "read from index metadata; nothing to configure here."
        )
        query = st.text_input(
            "Search query",
            placeholder="What is RAG?",
            key="query_input",
        )
        col_k, col_ex, col_rr = st.columns([3, 1, 1], vertical_alignment="bottom")
        with col_k:
            top_k = st.slider(
                "Chunks to return (top_k)",
                min_value=1,
                max_value=50,
                value=5,
                key="top_k_slider",
            )
        with col_ex:
            use_expand = st.checkbox(
                "Expand context",
                value=_opt_bool(ingest_opts, "search_expand", True),
                key="expand_checkbox",
                help="Return wider context (sentence window, hierarchical parent) when available.",
            )
        with col_rr:
            use_rerank = st.checkbox(
                "Rerank results",
                value=_opt_bool(ingest_opts, "rerank_enabled", False),
                key="rerank_checkbox",
            )

        if st.button(
            "Retrieve",
            icon=":material/search:",
            type="primary",
            disabled=not query.strip(),
            use_container_width=True,
            key="btn_retrieve",
        ):
            try:
                with st.spinner("POST /retrieve…"):
                    r = httpx.post(
                        f"{base}/retrieve",
                        json={
                            "query": query.strip(),
                            "top_k": int(top_k),
                            "index_id": index_id,
                            "rerank": use_rerank,
                            "expand": use_expand,
                        },
                        timeout=httpx.Timeout(120.0),
                    )
            except httpx.RequestError as e:
                st.error(f"Request failed: {e}")
            else:
                if r.status_code >= 400:
                    st.error(f"HTTP {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    chunks = data.get("chunks") or []
                    candidate_count = int(data.get("candidate_count") or 0)
                    pool_note = (
                        f" · **candidate pool:** {candidate_count}" if candidate_count else ""
                    )
                    st.markdown(
                        f"**index_id:** `{data.get('index_id', index_id)}` · "
                        f"**chunks:** {len(chunks)} · **query:** {data.get('query', '')!r}"
                        + pool_note
                    )
                    if not chunks:
                        st.info("No chunks. Ingest documents for this index or try a different query.")
                    else:
                        st.subheader("Results")
                    for i, ch in enumerate(chunks):
                        score = ch.get("score")
                        cid = ch.get("chunk_id", "")
                        meta = ch.get("metadata") or {}
                        title = (
                            f"[{i + 1}] {cid} — score {float(score):.4f}"
                            if isinstance(score, (int, float))
                            else f"[{i + 1}] {cid}"
                        )
                        with st.expander(title, expanded=(i == 0)):
                            if meta:
                                st.caption(f"metadata: {meta}")
                            st.write(ch.get("text", ""))

                    candidates = data.get("candidates") or []
                    if candidates:
                        with st.expander(
                            f"Candidate pool — {len(candidates)} chunk(s) before rerank/fusion (raw JSON)",
                            icon=":material/fact_check:",
                            expanded=False,
                        ):
                            st.caption(
                                "Assurance view, not results: the intermediate pool the final "
                                "chunks were selected from. Scores are pre-rerank."
                            )
                            st.json(candidates, expanded=False)


main()
