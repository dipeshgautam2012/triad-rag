"""Tier 1 RAG eval: loop golden.jsonl against /retrieve and /query, write CSV."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "generation"))

from app.ai_providers.provider_factory import make_provider  # noqa: E402


class GoldenRow(BaseModel):
    question: str
    ground_truth: str
    expected_source: str = ""
    expected_page: int | None = None


class FaithfulnessVerdict(BaseModel):
    faithful: bool
    reason: str = Field(default="", max_length=500)


def _call_api(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object from {url}")
    return data


def run_eval(
    rows: list[GoldenRow],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[float]]:
    retrieval_url = args.retrieval_url.rstrip("/")
    orchestrator_url = args.orchestrator_url.rstrip("/")
    report_rows: list[dict[str, Any]] = []
    hit_scores: list[float] = []
    overlap_scores: list[float] = []
    faith_scores: list[float] = []

    with httpx.Client(timeout=args.timeout_s) as client:
        for i, row in enumerate(rows, start=1):
            print(f"[{i}/{len(rows)}] {row.question[:80]}")
            record: dict[str, Any] = {
                "question": row.question,
                "expected_source": row.expected_source,
                "expected_page": row.expected_page or "",
                "hit_at_k": 0,
                "keyword_overlap": 0.0,
                "faithful": "",
                "faithfulness_reason": "",
                "answer": "",
                "notes": "",
                "error": "",
            }
            try:
                # --- Retrieval: POST /retrieve (port 8101) ---
                retrieve_payload = _call_api(
                    client,
                    f"{retrieval_url}/retrieve",
                    {"query": row.question, "top_k": args.top_k, "index_id": args.index_id},
                )
                chunks = retrieve_payload.get("chunks") or []

                # --- Evaluation: retrieval metrics (hit@k, keyword overlap) ---
                hit = False
                if row.expected_source:
                    expected = row.expected_source.strip()
                    for chunk in chunks:
                        meta = chunk.get("metadata") or {}
                        source = str(meta.get("source") or "")
                        if source != expected:
                            continue
                        if row.expected_page is None:
                            hit = True
                            break
                        page = meta.get("page")
                        try:
                            page = int(page) if page is not None else None
                        except (TypeError, ValueError):
                            page = None
                        if page == row.expected_page:
                            hit = True
                            break

                # --- Evaluation: keyword overlap ---
                words = {w.lower() for w in re.findall(r"[a-z0-9]+", row.ground_truth) if len(w) > 2}
                corpus = " ".join(str(c.get("text") or "") for c in chunks).lower()
                overlap = (sum(1 for w in words if w in corpus) / len(words)) if words else 0.0

                # --- Evaluation: hit@k ---
                record["hit_at_k"] = int(hit)
                record["keyword_overlap"] = round(overlap, 3)
                hit_scores.append(float(hit))
                overlap_scores.append(overlap)

                # --- Query: POST /query (port 8100; retrieve + generate) ---
                query_payload = _call_api(
                    client,
                    f"{orchestrator_url}/query",
                    {"question": row.question, "top_k": args.top_k, "index_id": args.index_id},
                )
                answer = str(query_payload.get("answer") or "").strip()
                sources = query_payload.get("sources") or []
                record["answer"] = answer[:500]

                # --- Evaluation: generation faithfulness (Gemini judge) ---
                if args.skip_faithfulness:
                    record["notes"] = "faithfulness skipped"
                elif not answer:
                    record["faithful"] = 0
                    record["faithfulness_reason"] = "empty answer"
                    faith_scores.append(0.0)
                else:
                    source_parts: list[str] = []
                    used = 0
                    for j, chunk in enumerate(sources, start=1):
                        meta = chunk.get("metadata") or {}
                        src = str(meta.get("source") or "unknown")
                        block = f"[{j}] ({src})\n{chunk.get('text', '')}\n"
                        if used + len(block) > 8000:
                            break
                        source_parts.append(block)
                        used += len(block)
                    sources_text = "".join(source_parts).strip() or "(no sources)"

                    prompt = (
                        "Evaluate whether the generated answer is fully supported by the retrieved sources.\n"
                        "Ignore whether it matches the reference answer; only check grounding in sources.\n\n"
                        f"Question:\n{row.question}\n\n"
                        f"Reference answer (for context only):\n{row.ground_truth}\n\n"
                        f"Generated answer:\n{answer}\n\n"
                        f"Retrieved sources:\n{sources_text}\n"
                    )
                    verdict_raw = make_provider().generate(
                        prompt, context="", output_schema=FaithfulnessVerdict
                    )
                    verdict = (
                        verdict_raw
                        if isinstance(verdict_raw, FaithfulnessVerdict)
                        else FaithfulnessVerdict.model_validate(verdict_raw)
                    )
                    record["faithful"] = int(verdict.faithful)
                    record["faithfulness_reason"] = verdict.reason
                    faith_scores.append(float(verdict.faithful))
            except Exception as e:
                record["error"] = str(e)[:300]
            report_rows.append(record)

    return report_rows, hit_scores, overlap_scores, faith_scores


def main(argv: list[str] | None = None) -> None:
    eval_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Run Tier 1 RAG evaluation against golden.jsonl.")
    p.add_argument("--golden", type=Path, default=eval_dir / "golden.jsonl", help="Path to golden.jsonl")
    p.add_argument("--output", type=Path, default=eval_dir / "report.csv", help="CSV report path")
    p.add_argument("--retrieval-url", default="http://127.0.0.1:8101")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:8100")
    p.add_argument("--index-id", default="default")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="skip Gemini faithfulness judge (retrieval metrics only)",
    )
    args = p.parse_args(argv)

    rows: list[GoldenRow] = []
    for line_no, line in enumerate(args.golden.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(GoldenRow.model_validate(json.loads(text)))
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"{args.golden}:{line_no}: invalid golden row: {e}") from e

    report_rows, hit_scores, overlap_scores, faith_scores = run_eval(rows, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "expected_source",
        "expected_page",
        "hit_at_k",
        "keyword_overlap",
        "faithful",
        "faithfulness_reason",
        "answer",
        "notes",
        "error",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    hit_avg = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
    overlap_avg = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    faith_avg = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    print(f"Wrote {args.output}")
    print(
        f"Summary: hit@{args.top_k}={hit_avg:.2%}, "
        f"keyword_overlap={overlap_avg:.2%}, "
        f"faithfulness={faith_avg:.2%}"
    )


if __name__ == "__main__":
    main()
