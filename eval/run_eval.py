"""Tier 1 RAG eval: loop golden.jsonl against /retrieve and /query, write versioned CSV."""

import argparse
import csv
import io
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
DEFAULT_DATASET_ID = "top_llm_questions"

_RETRYABLE_HTTP = frozenset({502, 503, 504})

REPORT_COLUMNS = [
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


class GoldenRow(BaseModel):
    question: str
    ground_truth: str
    expected_source: str = ""
    expected_page: int | None = None


class FaithfulnessVerdict(BaseModel):
    faithful: bool
    reason: str = Field(default="", max_length=500)


@dataclass
class EvalConfig:
    retrieval_url: str = "http://127.0.0.1:8101"
    orchestrator_url: str = "http://127.0.0.1:8100"
    generation_url: str = "http://127.0.0.1:8102"
    index_id: str = DEFAULT_DATASET_ID
    top_k: int = 5
    timeout_s: float = 120.0
    skip_faithfulness: bool = False
    retry_attempts: int = 3
    retry_wait_s: float = 0.5


ProgressCallback = Callable[[int, int, str], None]


def list_datasets() -> list[str]:
    if not DATASETS_DIR.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(DATASETS_DIR.iterdir()):
        if path.is_dir() and (path / "golden.jsonl").is_file():
            ids.append(path.name)
    return ids


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def golden_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "golden.jsonl"


def reports_dir(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "reports"


def new_report_run_dir(dataset_id: str, *, when: datetime | None = None) -> Path:
    ts = (when or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return reports_dir(dataset_id) / ts


def list_report_runs(dataset_id: str) -> list[Path]:
    root = reports_dir(dataset_id)
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def load_golden_rows(path: Path) -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(GoldenRow.model_validate(json.loads(text)))
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"{path}:{line_no}: invalid golden row: {e}") from e
    return rows


def load_golden_text(text: str, *, label: str = "golden") -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(GoldenRow.model_validate(json.loads(stripped)))
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"{label}:{line_no}: invalid golden row: {e}") from e
    return rows


def golden_expected_sources(rows: list[GoldenRow]) -> set[str]:
    return {row.expected_source.strip() for row in rows if row.expected_source.strip()}


def missing_golden_sources(rows: list[GoldenRow], corpus_files: set[str]) -> list[str]:
    return sorted(golden_expected_sources(rows) - corpus_files)


def _call_api(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    retry_attempts: int = 3,
    retry_wait_s: float = 0.5,
) -> dict[str, Any]:
    attempts = max(1, retry_attempts)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(retry_wait_s * attempt)
        try:
            r = client.post(url, json=payload)
            if r.status_code in _RETRYABLE_HTTP and attempt < attempts - 1:
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError(f"expected JSON object from {url}")
            return data
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_exc = e
            if attempt >= attempts - 1:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"request failed for {url}")


def select_generation_model(
    client: httpx.Client,
    orchestrator_url: str,
    provider: str,
    model_alias: str,
    *,
    retry_attempts: int = 3,
    retry_wait_s: float = 0.5,
) -> dict[str, Any]:
    return _call_api(
        client,
        f"{orchestrator_url.rstrip('/')}/models/select",
        {"provider": provider.strip(), "model_alias": model_alias.strip()},
        retry_attempts=retry_attempts,
        retry_wait_s=retry_wait_s,
    )


def _parse_faithfulness_json(text: str) -> FaithfulnessVerdict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return FaithfulnessVerdict.model_validate(data)


def _judge_faithfulness(
    client: httpx.Client,
    generation_url: str,
    evaluation_context: str,
    *,
    retry_attempts: int,
    retry_wait_s: float,
) -> FaithfulnessVerdict:
    judge_question = (
        "Evaluate whether the generated answer is fully supported by the retrieved sources. "
        "Ignore whether it matches the reference answer; only check grounding in sources. "
        'Respond with JSON only: {"faithful": true|false, "reason": "short explanation"}'
    )
    payload = _call_api(
        client,
        f"{generation_url.rstrip('/')}/generate",
        {"question": judge_question, "context": evaluation_context},
        retry_attempts=retry_attempts,
        retry_wait_s=retry_wait_s,
    )
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        raise ValueError("faithfulness judge returned empty answer")
    try:
        return _parse_faithfulness_json(answer)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"faithfulness judge returned non-JSON: {answer[:200]}") from None
        return _parse_faithfulness_json(match.group(0))


def run_eval(
    rows: list[GoldenRow],
    config: EvalConfig,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[float]]:
    retrieval_url = config.retrieval_url.rstrip("/")
    orchestrator_url = config.orchestrator_url.rstrip("/")
    report_rows: list[dict[str, Any]] = []
    hit_scores: list[float] = []
    overlap_scores: list[float] = []
    faith_scores: list[float] = []
    total = len(rows)
    api_kw = {
        "retry_attempts": config.retry_attempts,
        "retry_wait_s": config.retry_wait_s,
    }

    with httpx.Client(timeout=config.timeout_s) as client:
        for i, row in enumerate(rows, start=1):
            if on_progress is not None:
                on_progress(i, total, row.question)
            else:
                print(f"[{i}/{total}] {row.question[:80]}")

            record: dict[str, Any] = {
                "question": row.question,
                "expected_source": row.expected_source,
                "expected_page": "" if row.expected_page is None else str(row.expected_page),
                "hit_at_k": 0,
                "keyword_overlap": 0.0,
                "faithful": "",
                "faithfulness_reason": "",
                "answer": "",
                "notes": "",
                "error": "",
            }
            try:
                retrieve_payload = _call_api(
                    client,
                    f"{retrieval_url}/retrieve",
                    {"query": row.question, "top_k": config.top_k, "index_id": config.index_id},
                    **api_kw,
                )
                chunks = retrieve_payload.get("chunks") or []

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

                words = {w.lower() for w in re.findall(r"[a-z0-9]+", row.ground_truth) if len(w) > 2}
                corpus = " ".join(str(c.get("text") or "") for c in chunks).lower()
                overlap = (sum(1 for w in words if w in corpus) / len(words)) if words else 0.0

                record["hit_at_k"] = int(hit)
                record["keyword_overlap"] = round(overlap, 3)
                hit_scores.append(float(hit))
                overlap_scores.append(overlap)

                query_payload = _call_api(
                    client,
                    f"{orchestrator_url}/query",
                    {"question": row.question, "top_k": config.top_k, "index_id": config.index_id},
                    **api_kw,
                )
                answer = str(query_payload.get("answer") or "").strip()
                sources = query_payload.get("sources") or []
                record["answer"] = answer[:500]

                if config.skip_faithfulness:
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
                        f"Question:\n{row.question}\n\n"
                        f"Reference answer (for context only):\n{row.ground_truth}\n\n"
                        f"Generated answer:\n{answer}\n\n"
                        f"Retrieved sources:\n{sources_text}\n"
                    )
                    try:
                        verdict = _judge_faithfulness(
                            client,
                            config.generation_url,
                            prompt,
                            retry_attempts=config.retry_attempts,
                            retry_wait_s=config.retry_wait_s,
                        )
                    except ValueError as e:
                        record["notes"] = f"faithfulness skipped: {e}"[:200]
                    else:
                        record["faithful"] = int(verdict.faithful)
                        record["faithfulness_reason"] = verdict.reason
                        faith_scores.append(float(verdict.faithful))
            except Exception as e:
                record["error"] = str(e)[:300]
            report_rows.append(record)

    return report_rows, hit_scores, overlap_scores, faith_scores


def summarize_metrics(
    hit_scores: list[float],
    overlap_scores: list[float],
    faith_scores: list[float],
    *,
    top_k: int,
) -> dict[str, float | int]:
    hit_avg = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
    overlap_avg = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    faith_avg = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    return {
        "questions": len(hit_scores),
        "top_k": top_k,
        "hit_at_k": hit_avg,
        "keyword_overlap": overlap_avg,
        "faithfulness": faith_avg,
    }


def write_report_csv(path: Path, report_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(report_rows)


def write_report_run(
    dataset_id: str,
    report_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    when: datetime | None = None,
) -> Path:
    run_dir = new_report_run_dir(dataset_id, when=when)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.csv"
    write_report_csv(report_path, report_rows)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def report_to_csv_text(report_rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(report_rows)
    return buf.getvalue()


def load_report_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run Tier 1 RAG evaluation against a dataset golden.jsonl.")
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help=f"Dataset folder under eval/datasets/ (default: {DEFAULT_DATASET_ID})",
    )
    p.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Override golden.jsonl path (default: eval/datasets/<dataset>/golden.jsonl)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override report.csv path (default: new timestamped run under dataset/reports/)",
    )
    p.add_argument("--retrieval-url", default="http://127.0.0.1:8101")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:8100")
    p.add_argument("--generation-url", default="http://127.0.0.1:8102")
    p.add_argument("--index-id", default=None, help="defaults to --dataset id")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="skip LLM faithfulness judge (retrieval metrics + answers only)",
    )
    args = p.parse_args(argv)

    dataset_id = args.dataset.strip()
    golden = args.golden or golden_path(dataset_id)
    rows = load_golden_rows(golden)
    index_id = (args.index_id or "").strip() or dataset_id
    config = EvalConfig(
        retrieval_url=args.retrieval_url,
        orchestrator_url=args.orchestrator_url,
        generation_url=args.generation_url,
        index_id=index_id,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        skip_faithfulness=args.skip_faithfulness,
    )
    report_rows, hit_scores, overlap_scores, faith_scores = run_eval(rows, config)

    summary = summarize_metrics(hit_scores, overlap_scores, faith_scores, top_k=config.top_k)
    run_meta = {
        "dataset_id": dataset_id,
        "index_id": index_id,
        "top_k": config.top_k,
        "skip_faithfulness": config.skip_faithfulness,
        "metrics": summary,
    }
    if args.output is not None:
        out_path = args.output
        write_report_csv(out_path, report_rows)
    else:
        out_path = write_report_run(dataset_id, report_rows, run_meta)

    print(f"Wrote {out_path}")
    print(
        f"Summary: hit@{summary['top_k']}={summary['hit_at_k']:.2%}, "
        f"keyword_overlap={summary['keyword_overlap']:.2%}, "
        f"faithfulness={summary['faithfulness']:.2%}"
    )


if __name__ == "__main__":
    main()
