"""Corpus folder — list and delete raw .txt / .pdf files on disk."""

from pathlib import Path

from app.indexers.base_indexer import validate_index_id


def corpus_dir(corpus_root: Path, index_id: str) -> Path:
    validate_index_id(index_id)
    root = corpus_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if index_id == "default":
        return root
    sub = (root / index_id).resolve()
    try:
        sub.relative_to(root)
    except ValueError as e:
        raise ValueError("invalid index_id path") from e
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def list_corpus_files(corpus_root: Path, index_id: str) -> list[str]:
    corpus = corpus_dir(corpus_root, index_id)
    names = {p.name for p in [*corpus.glob("*.txt"), *corpus.glob("*.pdf")]}
    return sorted(names)


def unlink_corpus_file(corpus_root: Path, index_id: str, filename: str) -> str:
    name = Path(filename).name
    corpus = corpus_dir(corpus_root, index_id)
    dest = (corpus / name).resolve()
    try:
        dest.relative_to(corpus.resolve())
    except ValueError as e:
        raise ValueError("Invalid path") from e
    if not dest.is_file():
        raise FileNotFoundError(name)

    dest.unlink()
    if not list_corpus_files(corpus_root, index_id) and index_id != "default":
        try:
            corpus.rmdir()
        except OSError:
            pass
    return name
