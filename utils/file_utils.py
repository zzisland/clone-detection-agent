from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def ensure_repo_path(repo_path: str) -> Path:
    p = Path(repo_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"repo_path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"repo_path is not a directory: {p}")
    return p


def iter_source_files(repo_path: Path, patterns: Iterable[str] = ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp")) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(repo_path.rglob(pattern))
    return sorted({f for f in files if f.is_file()})


def to_posix_relative(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()
