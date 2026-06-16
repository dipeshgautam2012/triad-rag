"""Save an uploaded file into the corpus folder (POST /ingest)."""

import logging
from pathlib import Path

from app.ingest.filename_sanitizer import sanitize_corpus_filename

logger = logging.getLogger(__name__)


def save_upload(
    data: bytes,
    original_filename: str | None,
    corpus_dir: Path,
    *,
    max_bytes: int,
) -> Path:
    """Write bytes to corpus_dir; enforces max_bytes and filename sanitization."""
    if len(data) > max_bytes:
        raise ValueError(f"File too large (max {max_bytes} bytes)")

    name = sanitize_corpus_filename(original_filename)
    corpus_dir = corpus_dir.resolve()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    dest = (corpus_dir / name).resolve()
    try:
        dest.relative_to(corpus_dir)
    except ValueError as e:
        raise ValueError("Invalid path") from e

    dest.write_bytes(data)
    logger.info("Ingested corpus file %s (%s bytes)", dest.name, len(data))
    return dest
