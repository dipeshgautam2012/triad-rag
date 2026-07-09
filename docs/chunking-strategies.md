# Chunking strategies

Chunking splits uploaded documents into smaller passages before indexing. You pick a chunker **once** when you first upload to a collection; later uploads to the same collection must use the same chunker.

All chunkers live in `retrieval/app/chunkers/` and use **LlamaIndex** node parsers. PDFs are read with **pypdf** (one page at a time); plain text files are read as a single document.

**Naming:** the **section-based** chunker splits on document structure (`#` headings). In config, API, and UI it is selected as **`markdown`** (implementation uses LlamaIndex `MarkdownNodeParser`). It only works on **markdown text** — not on PDF or other formats unless you convert them to markdown first (see [Section-based (`markdown`)](#section-based-markdown)).

**Chunking vs search:** chunking decides *how* a file is cut into passages. **Retrieval** (chroma, BM25, or hybrid — see [`retrieval-strategies.md`](retrieval-strategies.md)) decides *how* your question is matched against those passages. This doc covers both: how chunks are built and how search uses them.

---

## How retrieval uses your chunks

Every chunker outputs a **`ChunkSet`** with two lists:

| List | Meaning | Used at ingest | Used at search |
|------|---------|----------------|----------------|
| **`embed_chunks`** | Passages that are **searchable** | Embedded into Chroma (vector/hybrid) and/or BM25 (bm25/hybrid) | **Always** — every indexer matches the question against these |
| **`all_chunks`** | Full tree when some nodes are **not** embedded (e.g. parent sections) | Saved to the **node store** only — not searched directly | Lookup only — chroma/hybrid may **expand** child hits to a parent passage |

```
Ingest
  file → chunker → ChunkSet
                      ├── embed_chunks ──► Chroma (vectors) and/or BM25 (keywords)
                      └── all_chunks? ──► node store (parents / extra nodes)

Query (POST /retrieve)
  question
      → search embed_chunks (by meaning, keywords, or both — collection indexer)
      → optional: merge child hits → parent text (vector/hybrid + node store)
      → optional: rerank (wider pool → cross-encoder → top_k)
      → chunks[] returned to UI
```

**What comes back:** each hit is a `RetrievedChunk` with `chunk_id`, `text`, `score`, and `metadata` (`source`, `page`, etc.). The **`text`** field is what the Index UI and Chat UI show as the passage. Which string that is depends on the chunker (see **At retrieve time** under each strategy below).

The collection’s **indexer** (`chroma`, `bm25`, `hybrid`) does not change how chunks are cut — it only changes *how* `embed_chunks` are scored. Rerank and expand are separate per-query options (defaults in `env.toml`: `rerank_enabled`, `search_expand`).

---

## Quick comparison

| Chunker | Best for | Searchable unit (`embed_chunks`) | What `text` in results usually is |
|---------|----------|----------------------------------|-----------------------------------|
| **simple** | General use, default | Every chunk | Same fixed-size passage that was embedded |
| **section-based** (`markdown`) | Markdown `.txt` with `#` headings only | Every chunk (section-first, then size) | Same section passage that was embedded |
| **hierarchical** | Small hits + wider context | Child chunks only | Child passage, or **parent** text if several children from same parent match |
| **sentence_window** | Precise sentence matches | Anchor sentence per node | Window in `metadata`; `/retrieve` `text` uses window |
| **semantic** | Topic-based boundaries | Topic-sized chunks | Same topic-sized passage that was embedded |

| Chunker | Needs embeddings at chunk time? |
|---------|--------------------------------|
| **simple**, **hierarchical**, **section-based** (`markdown`), **sentence_window** | No |
| **semantic** | **Yes** (same model as ingest) |

**Libraries**

| Piece | Library |
|-------|---------|
| File reading (PDF) | `pypdf` |
| All chunk parsers | `llama_index.core` (`SentenceSplitter`, `MarkdownNodeParser`, `HierarchicalNodeParser`, `SentenceWindowNodeParser`, `SemanticSplitterNodeParser`) |
| Semantic chunking model | `llama_index.embeddings.huggingface` (via your chosen embedding model) |

---

## simple

Splits text into fixed-size pieces with a small overlap so sentences are not cut off mid-thought.

**LlamaIndex:** `SentenceSplitter`

**Example** — `chunk_size=100`, `chunk_overlap=20`:

```
Input:  "Our refund policy allows returns within 30 days. Contact support with your order ID. Shipping fees are non-refundable."

Chunk 0: "Our refund policy allows returns within 30 days. Contact support with your order ID."
Chunk 1: "Contact support with your order ID. Shipping fees are non-refundable."
         ↑ overlap keeps shared context
```

Every chunk is embedded and searched. Works well for most PDFs and `.txt` files.

**At retrieve time:** vector search compares the question embedding to each chunk embedding; BM25 matches question words against the same chunk `text`. Each hit is one chunk — no parent/child merge (`all_chunks` is `None`).

---

## Section-based (`markdown`)

Section-based chunking splits on **document headings** (`#`, `##`, …) so each chunk stays inside one section. Oversized sections are sub-split with `chunk_size` / `chunk_overlap`; overlap does not cross section boundaries.

> **Markdown only.** This chunker parses **markdown syntax** (`#` headings, etc.). It does **not** read PDF structure, Word outlines, or HTML tags. Upload **markdown text** (today: a `.txt` file whose content uses `#` headings). For **PDF** or any other format, convert to markdown **before** ingest (e.g. with Docling, pymupdf4llm, or LlamaParse) — or use **simple** / **hierarchical** on the raw file instead.

**Chunker id:** `markdown` (Index UI, `chunker_name` on ingest, index metadata)

**LlamaIndex:** `MarkdownNodeParser`, then `SentenceSplitter` for long sections

**What ingest accepts today:** `.txt` and `.pdf` only. For this chunker, the file content must already be markdown. A `.pdf` uploaded as-is is loaded with **pypdf** as flat page text (no `#` lines) — section splitting will not work until you convert the PDF to markdown offline and upload the result as `.txt`.

**Example** (markdown in a `.txt` file):

```
# Refund policy
Returns within 30 days. Contact support with your order ID.

## Shipping
Fees are non-refundable.
```

→ two section chunks (or more if a section exceeds `chunk_size`). Metadata includes `header_path` (e.g. `/Refund policy/` or `/Refund policy/Shipping/`).

**At retrieve time:** same as **simple** — every chunk is searchable; no parent merge. `header_path` is returned in chunk metadata for citations.

---

## hierarchical

Two-level **parent + child** storage:

- **Children** (small) → embedded and searched
- **Parents** (larger, `chunk_size × parent_multiplier`) → saved in the **node store**, not embedded

At search time, if several children from the same parent match, the system can **merge up** to the wider parent passage (LlamaIndex `AutoMergingRetriever`).

**LlamaIndex:** `HierarchicalNodeParser`, `AutoMergingRetriever`

**Example:**

```
Parent:  "Chapter 3 — Refunds. Full policy text spanning two pages…"
           ├── child 3a: "Refunds within 30 days…"     ← search hit
           └── child 3b: "Contact support with ID…"   ← search hit
```

If both children score well, the user may see the full parent text instead of two tiny snippets.

Chunk metadata includes `chunk_role` (`parent` / `child`) and `parent_id` on children.

Config: `hierarchical_parent_multiplier` (how much larger parents are than children).

**At retrieve time:** search runs on **children** (small, precise hits); parents live in the node store. On **chroma/hybrid** with **`expand: true`** (or `search_expand` default), chroma may auto-merge children into a parent when **`ratio > 0.4`**, then widen text via the node store. BM25 uses node-store expansion only (no auto-merge). Details: [Parent merge](#parent-merge-auto-merge).

---

## sentence_window

LlamaIndex `SentenceWindowNodeParser`: one **anchor sentence** per searchable node; a paired **window** node holds surrounding sentences in the node store.

`window_size` = sentences on **each side** of the anchor ([LlamaIndex docs](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)).

- **`embed_chunks`:** anchor sentences (vector + BM25)
- **`all_chunks`:** window passages (node store only)
- Anchor metadata: `context_node_id` → window node id

**Search** matches the anchor. **`/retrieve` `text`** resolves the window from the node store via `passage_text()` in the indexers (not chunker-specific code in `main`).

Config: `sentence_window_size`. Re-ingest collections built with the old window-in-index behavior.

---

## semantic

Does **not** use a fixed character count. It walks the document, compares meaning between sentences, and starts a **new chunk when the topic changes**.

**LlamaIndex:** `SemanticSplitterNodeParser` + your embedding model

**Example:**

```
Paragraph about refunds  → chunk 1
Paragraph about shipping → chunk 2  (split because topic shifted)
Paragraph about warranties → chunk 3
```

Because it calls the embedding model during ingest, **semantic** requires an embedding model even on **bm25-only** collections.

Config: `semantic_breakpoint_percentile`, `semantic_buffer_size`.

**At retrieve time:** same as **simple** — each topic-sized chunk is one searchable unit. Boundaries were chosen at ingest by meaning shift, not character count; search does not re-split.

---

## What gets stored

Every chunker attaches the same base metadata:

| Field | Meaning |
|-------|---------|
| `source` | Filename |
| `file_type` | `txt` or `pdf` |
| `page` | PDF page number (if applicable) |

Chunkers may add `chunk_role`, `parent_id`, `header_path`, `window`, or `original_text` as described above.

The chosen chunker name is saved on the collection as `chunker` in index metadata and cannot be changed on re-ingest.

### Storage vs search (summary)

| Chunker | In Chroma / BM25 (`embed_chunks`) | In node store (`all_chunks`) |
|---------|-----------------------------------|------------------------------|
| **simple** | All chunks | — |
| **section-based** (`markdown`) | All chunks | — |
| **hierarchical** | Children only | Parents + children |
| **sentence_window** | All chunks (anchor sentence) | — |
| **semantic** | All chunks | — |

At query time, **BM25 and chroma always start from `embed_chunks`**. With **`expand`**, hits can be widened from the node store (`context_node_id`, sentence window, etc.). Hierarchical **parent auto-merge** (`ratio > 0.4`) runs on the **chroma leg** only — see **Parent merge** below.

### Parent merge (auto-merge)

Applies to the **hierarchical** chunker on **chroma** or **hybrid** collections when **`expand`** is on. The BM25 leg in hybrid does not auto-merge siblings into a parent (it may still use node-store text expansion).

**What is `ratio`?**

**`ratio`** is the number of children hit by search, out of the total number of children of that parent.

Example: a parent has 10 children; chroma search returns 5 of them → ratio is 5 out of 10, or **0.5**.

**Merge when `ratio > 0.4`** — replace those child hits with one parent passage (`text` = parent content; score = average of the merged children). Exactly 0.4 does **not** merge. The cutoff `0.4` is `simple_ratio_thresh`, **hardcoded** in `chroma_indexer.py` (not in `env.toml`). Implemented by LlamaIndex `AutoMergingRetriever` when `expand` is true.

After that, `_expand_hit_context` in `base_indexer.py` may still replace text using `context_node_id` from the node store (sentence window, etc.).

**When merge can run at all (all must be true):**

| Requirement | Where it is set |
|-------------|-----------------|
| Collection indexer is `chroma` or `hybrid` | Chosen at first upload |
| Chunker is **hierarchical** (node store has parent nodes) | Chosen at first upload |
| `expand: true` on retrieve, or `search_expand = true` in `env.toml` | Per query or default |
| Node store exists for the collection | Built at ingest when `all_chunks` is not `None` |

**Examples:**

| Total children | Children in hit list | ratio | `ratio > 0.4`? |
|----------------|---------------------|-------|----------------|
| 10 | 5 | 5 out of 10 (0.50) | Yes → merge |
| 10 | 4 | 4 out of 10 (0.40) | No |
| 5 | 3 | 3 out of 5 (0.60) | Yes → merge |
| 2 | 1 | 1 out of 2 (0.50) | Yes → merge |

**Hybrid specifically:** parent auto-merge runs on the **chroma leg only**, before RRF. BM25 hits stay as individual leaves unless node-store expansion applies. The fused list can mix parent passages (from chroma) with child passages (from BM25).

**Tuning retrieval pool size (not the 0.4 cutoff):** `hybrid_candidate_multiplier` and `rerank_candidate_multiplier` in `env.toml` control how many hits are fetched *before* fusion/rerank — a wider pool makes it more likely several siblings land in the chroma hit list, but the rule itself stays **`ratio > 0.4`**.

---

## Choosing a chunker

| Situation | Reasonable choice |
|-----------|-------------------|
| Not sure / mixed documents | **simple** |
| Markdown `.txt` with `#` headings (policies, README-style docs) | **section-based** (`markdown`) |
| PDF or other formats without markdown conversion | **not** section-based — use **simple** / **hierarchical**, or convert to markdown first |
| Small search hits + optional parent merge on PDFs / long docs | **hierarchical** |
| Short Q&A, precise sentences | **sentence_window** |
| Topics change sharply within pages | **semantic** |

See also: [`retrieval-strategies.md`](retrieval-strategies.md) (chroma / BM25 / hybrid / rerank / expand) · [`DESIGN.md`](DESIGN.md) · [`../README.md`](../README.md)
