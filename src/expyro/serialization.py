import pickle
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def dump(obj: T, path: Path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load(path: Path) -> T:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.parent}")

    with open(path, "rb") as file:
        return pickle.load(file)


def dump_config(config: T, folder: Path):
    assert folder.exists()
    dump(config, folder / "config.pkl")


def dump_result(result: T, folder: Path):
    assert folder.exists()
    dump(result, folder / "result.pkl")


def load_config(folder: Path) -> T:
    assert folder.is_dir()
    return load(folder / "config.pkl")


def load_result(folder: Path) -> T:
    assert folder.is_dir()
    return load(folder / "result.pkl")


def has_config(folder: Path) -> bool:
    return (folder / "config.pkl").exists()


def has_result(folder: Path) -> bool:
    return (folder / "result.pkl").exists()
