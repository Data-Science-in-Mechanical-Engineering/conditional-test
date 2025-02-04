from __future__ import annotations

from pathlib import Path


def unique_new_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    new_path = parent / f"{stem} ({counter}){suffix}"

    while new_path.exists():
        counter += 1
        new_path = parent / f"{stem} ({counter}){suffix}"

    return new_path
