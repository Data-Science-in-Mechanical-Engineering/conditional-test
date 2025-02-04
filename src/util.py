from pathlib import Path

from typing_extensions import NamedTuple

from src.expyro.experiment import Run
from src.rkhs import Kernel

ROOT_FOLDER = Path(__file__).parent.parent
DIR_RESULTS = ROOT_FOLDER / "results"
DIR_FIGURES = ROOT_FOLDER / "figures"
DIR_FIGURES_APX = DIR_FIGURES / "apx"

DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_FIGURES.mkdir(parents=True, exist_ok=True)
DIR_FIGURES_APX.mkdir(parents=True, exist_ok=True)


class KernelParametrization[T1: Kernel, T2: Kernel](NamedTuple):
    x: T1
    y: T2
    regularization: float


def move_experiment_run(run: Run, sub_dir: str | None, dir_name: str):
    if sub_dir is None:
        new_parent_dir = run.location.parent
    else:
        new_parent_dir = run.location.parent / sub_dir

    new_parent_dir.mkdir(parents=True, exist_ok=True)
    run.location.rename(new_parent_dir / dir_name)
