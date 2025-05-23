from collections.abc import Callable, Hashable
from typing import NamedTuple, Self, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import Array
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.experiments.monitoring.__main__ import experiment
from src.experiments.monitoring.spec import Config, MultipleResult
from src.expyro.experiment import Run
from src.figures.util import make_color_palette, make_line_style_palette, MulticolorPatch, MulticolorPatchHandler
from src.util import set_plot_style, FIGURE_WIDTH_COLUMN, TEXT_FONT_SIZE, COLOR_GRID, DIR_FIGURES

FIGURE_HEIGHT = 2.75

DIMENSION_DISTURBANCE = 0.1
DISTURBANCE_DIMENSION = 16
DIMENSIONS = [2, 4, 8, 16]
DISTURBANCES = [0.5, 0.75]


def extract_shared_parameter[T](runs: list[Run[Config, MultipleResult]], parameter: Callable[[Config], T]) -> T:
    parameters = {parameter(run.config) for run in runs}
    assert len(parameters) == 1
    return parameters.pop()


class Interval(NamedTuple):
    mean: Array
    lo: Array
    hi: Array

    @classmethod
    def from_runs(
            cls,
            runs: list[Run[Config, MultipleResult]],
            statistic: Callable[[Run[Config, MultipleResult]], Array]
    ) -> Self:
        significance_level = extract_shared_parameter(runs, lambda config: config.test.significance_level)

        # for every random function: for each dataset: for each time step: the statistic at that time step
        # shape: (n_functions, n_repetitions, n_timesteps)
        statistics = jnp.stack([statistic(run) for run in runs])

        # take the mean and confidence interval over all repetitions
        mean_statistic = statistics.mean(axis=(0, 1))
        lo_statistic = jnp.quantile(statistics, q=significance_level, axis=(0, 1))
        hi_statistic = jnp.quantile(statistics, q=1 - significance_level, axis=(0, 1))

        return cls(
            mean=mean_statistic,
            lo=lo_statistic,
            hi=hi_statistic
        )

    def __len__(self) -> int:
        assert self.mean.ndim == 1
        assert self.mean.shape == self.lo.shape == self.hi.shape
        return self.mean.shape[0]


def draw_box(x0: float, y0: float, width: float, height: float, arrow_x1: float, text: str, ax: plt.Axes):
    box = FancyBboxPatch(
        (x0, y0 - height / 2),
        width,
        height,
        boxstyle="round, rounding_size=2.5, pad=0",
        mutation_aspect=0.05,
        mutation_scale=1,
        linewidth=1,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(box)

    ax.text(
        x0 + width / 2,
        y0,
        text,
        ha="center",
        va="center",
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        color="black",
    )

    ax.annotate(
        "",
        xy=(arrow_x1, y0),
        xytext=(x0 + width, y0),
        zorder=10,
        arrowprops=dict(
            arrowstyle="->, head_width=0.2, head_length=0.2",
            lw=1,
            color="black",
            shrinkA=0,
            shrinkB=0,
        )
    )


def plot[T: Hashable](
        run_groups: dict[T, list[Run[Config, MultipleResult]]],
        colors: dict[T, str],
        line_styles: dict[T, str],
        line_width: float,
        ci: bool,
        ax_ratio: plt.Axes,
        ax_std: plt.Axes
):
    all_runs = [run for runs in run_groups.values() for run in runs]
    t_change = extract_shared_parameter(all_runs, lambda config: config.t_change)
    t_adapted = extract_shared_parameter(all_runs, lambda config: config.t_adapted)

    def _plot(ax_: plt.Axes, statistic: Callable[[Run[Config, MultipleResult]], Array], ci_: bool):
        for dimension, runs in run_groups.items():
            interval = Interval.from_runs(runs, statistic)

            time = jnp.arange(len(interval))

            sns.lineplot(
                x=time, y=interval.mean,
                color=colors[dimension], linestyle=line_styles[dimension], linewidth=line_width,
                zorder=10,
                ax=ax_
            )

            if ci_:
                ax_.fill_between(
                    x=time, y1=interval.lo, y2=interval.hi,
                    color=colors[dimension], alpha=0.1, linewidth=0,
                )

        ax_.axvline(x=t_change, color=COLOR_GRID, linestyle="dotted", linewidth=0.5, zorder=9)
        ax_.axvline(x=t_adapted, color=COLOR_GRID, linestyle="dotted", linewidth=0.5, zorder=9)
        ax_.axhline(y=1, color=COLOR_GRID, linestyle="dotted", linewidth=0.5, zorder=9)

        ax_.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
        ax_.minorticks_on()

        ax_.set_xmargin(0)

    def max_ratio(run: Run[Config, MultipleResult]) -> Array:
        return run.result.max_ratios()

    def normalized_mean_reference_std(run: Run[Config, MultipleResult]) -> Array:
        mean_reference_std = run.result.reference_mean_std()
        return mean_reference_std / mean_reference_std[:, [0]]

    _plot(ax_ratio, max_ratio, ci)
    _plot(ax_std, normalized_mean_reference_std, False)

    draw_box(t_adapted - 165, 3.50, 90, 0.85, t_change, "Change occurs", ax_ratio)
    draw_box(t_adapted - 165, 1.95, 90, 1.50, t_adapted, "Window done\nadapting", ax_ratio)

    ax_ratio.set_ylabel(r"Max. ratio")
    ax_ratio.set_ylim(bottom=0, top=5)

    ax_std.set_ylabel(r"Norm. std.")
    ax_std.set_xlabel(r"Time")

    ax_ratio.yaxis.set_major_locator(MultipleLocator(1))

    for ax in [ax_ratio, ax_std]:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def main():
    fig, [ax_ratio, ax_std] = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=False,
        figsize=(FIGURE_WIDTH_COLUMN, FIGURE_HEIGHT),
        height_ratios=[0.6, 0.4]
    )

    base_path = experiment.directory / experiment.name

    runs_by_dimension = {
        dimension: [
            experiment[path]
            for path in (base_path / f"d={dimension}__disturbance={DIMENSION_DISTURBANCE}").iterdir()
            if path.is_dir()
        ]
        for dimension in DIMENSIONS
    }

    runs_by_disturbance = {
        disturbance: [
            experiment[path]
            for path in (base_path / f"d={DISTURBANCE_DIMENSION}__disturbance={disturbance}").iterdir()
            if path.is_dir()
        ]
        for disturbance in DISTURBANCES
    }

    colors_dimension = make_color_palette("bright", DIMENSIONS)
    colors_disturbance = {disturbance: colors_dimension[DISTURBANCE_DIMENSION] for disturbance in DISTURBANCES}

    line_styles_dimension = {dimension: "-" for dimension in DIMENSIONS}
    line_styles_disturbance = make_line_style_palette(DISTURBANCES, styles=["--", "-.", ":"])

    plot(
        run_groups=runs_by_dimension,
        colors=colors_dimension,
        line_styles=line_styles_dimension,
        line_width=2,
        ci=True,
        ax_ratio=ax_ratio,
        ax_std=ax_std
    )

    plot(
        run_groups=runs_by_disturbance,
        colors=colors_disturbance,
        line_styles=line_styles_disturbance,
        line_width=1,
        ci=False,
        ax_ratio=ax_ratio,
        ax_std=ax_std
    )

    def handle_dimension(dimension: int) -> Line2D:
        return plt.Line2D(
            [], [],
            color=colors_dimension[dimension],
            linestyle=line_styles_dimension[dimension],
            linewidth=2
        )

    def handle_disturbance(disturbance: float) -> Line2D:
        return plt.Line2D(
            [], [],
            color=colors_disturbance[disturbance],
            linestyle=line_styles_disturbance[disturbance],
        )

    assert len(DIMENSIONS) == 4
    assert len(DISTURBANCES) == 2

    dummy_handle = Line2D([], [], color="white", linestyle="-", linewidth=0)

    dimension_disturbance_patch = MulticolorPatch(
        colors=[colors_dimension[dimension] for dimension in DIMENSIONS],
        linewidth=2,
        line_style="-",
        round=True
    )

    handles: list[Any] = [dimension_disturbance_patch]
    labels = [fr"$\xi={DIMENSION_DISTURBANCE:.2f}$"]

    handles.extend((handle_dimension(dimension) for dimension in DIMENSIONS))
    labels.extend((fr"$d={dimension}$" for dimension in DIMENSIONS))

    handles.append(dummy_handle)
    labels.append("")
    handles.append(handle_disturbance(DISTURBANCES[0]))
    labels.append(fr"$\xi={DISTURBANCES[0]:.2f}$")
    handles.append(handle_disturbance(DISTURBANCES[1]))
    labels.append(fr"$\xi={DISTURBANCES[1]:.2f}$")

    fig.legend(
        handles=handles,
        labels=labels,
        loc="center right",
        ncol=1,
        facecolor="white",
        frameon=False,
        fontsize=TEXT_FONT_SIZE,
        bbox_to_anchor=(1, 0, 0, 1.1),
        labelspacing=1.1,
        handler_map={
            MulticolorPatch: MulticolorPatchHandler()
        }
    )

    ax_ratio.set_xticks(ax_ratio.get_xticks()[1:-1])

    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(right=0.8)

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIR_FIGURES / "monitoring.pdf", dpi=500)
    plt.savefig(DIR_FIGURES / "monitoring.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    set_plot_style("white")
    main()
