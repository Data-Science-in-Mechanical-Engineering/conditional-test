from pathlib import Path
from typing import Generator, Literal

import jax
import matplotlib
import seaborn as sns

from src.expyro.experiment import Run

DIR_ROOT = Path(__file__).parent.parent
DIR_RESULTS = DIR_ROOT / "results"
DIR_FIGURES = DIR_ROOT / "figures"


def generate_random_keys(seed: int) -> Generator[jax.Array, None, None]:
    key = jax.random.PRNGKey(seed)

    while True:
        key, subkey = jax.random.split(key)
        yield subkey


TEXT_FONT_SIZE = 10  # pt
FIGURE_WIDTH_COLUMN = 5.4  # in
FIGURE_WIDTH_FULL = 6.75  # in

COLOR_GRID = "#cfcfcf"


def set_plot_style(grid: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid"):
    sns.set_style(grid)
    matplotlib.rcParams.update({"font.size": TEXT_FONT_SIZE})
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["grid.color"] = COLOR_GRID
    matplotlib.rcParams["axes.edgecolor"] = COLOR_GRID
    matplotlib.rcParams["grid.linewidth"] = 0.5
    matplotlib.rcParams["xtick.color"] = COLOR_GRID
    matplotlib.rcParams["ytick.color"] = COLOR_GRID
    matplotlib.rcParams["xtick.labelcolor"] = "black"
    matplotlib.rcParams["ytick.labelcolor"] = "black"
    matplotlib.rcParams["lines.dash_capstyle"] = "round"
    matplotlib.rcParams["lines.solid_capstyle"] = "round"


def move_experiment_run(run: Run, sub_dir: str | None, dir_name: str, skip_exists: bool = False):
    if sub_dir is None:
        new_parent_dir = run.location.parent
    else:
        new_parent_dir = run.location.parent / sub_dir

    new_parent_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    unique_dir_name = dir_name
    while (new_parent_dir / unique_dir_name).exists():
        unique_dir_name = f"{dir_name} ({i})"
        i += 1

    if skip_exists and i != 1:
        return

    run.location.rename(new_parent_dir / unique_dir_name)
