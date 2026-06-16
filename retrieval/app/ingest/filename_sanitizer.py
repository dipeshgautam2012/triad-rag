"""Make upload filenames safe (.txt and .pdf only)."""

import re
from pathlib import Path

_ALLOWED_SUFFIXES = frozenset({".txt", ".pdf"})
_STEM_SANITIZE = re.compile(r"[^a-zA-Z0-9._\-]+")


def sanitize_corpus_filename(original: str | None) -> str:
    """Return a cleaned basename or raise if the name is not allowed."""
    raw = (original or "").strip()
    base = Path(raw).name
    if not base:
        raise ValueError("Missing filename")
    suffix = Path(base).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(f"Only {sorted(_ALLOWED_SUFFIXES)} files are allowed")

    stem = _STEM_SANITIZE.sub("_", Path(base).stem).strip("._-")
    if not stem:
        stem = "upload"
    if len(stem) > 180:
        stem = stem[:180]

    base = f"{stem}{suffix}"
    if len(base) > 200:
        raise ValueError("Filename too long after sanitization")
    return base
