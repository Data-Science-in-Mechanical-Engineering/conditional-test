import jax.numpy as jnp
import seaborn as sns
from jax import Array
from matplotlib import pyplot as plt, patches

from src.experiments.example_1d.__main__ import experiment
from src.experiments.example_1d.spec import Result, Config
from src.expyro.experiment import Run
from src.figures.util import MulticolorPatch, MulticolorPatchHandler
from src.rkhs import RKHSFn
from src.rkhs.testing import ConditionalTestEmbedding, ConditionedTestEmbedding
from src.util import set_plot_style, FIGURE_WIDTH_COLUMN, TEXT_FONT_SIZE, DIR_FIGURES, COLOR_GRID

FIGURE_HEIGHT = 1.5


def plot_estimate(
        fn: RKHSFn, cme: ConditionalTestEmbedding, kmes: ConditionedTestEmbedding, state_space: Array,
        significance_level: float, color: str, ax: plt.Axes
) -> None:
    fn_values = fn(state_space)
    estimated_values = jnp.einsum("ni,ni->n", kmes.kme.coefficients, kmes.kme.points)
    states = state_space.reshape(-1)

    sns.lineplot(
        x=states, y=fn_values,
        color=color, linestyle="--", linewidth=1,
        estimator=None,
        ax=ax
    )

    sns.lineplot(
        x=states, y=estimated_values,
        color=color, linestyle="-", linewidth=2,
        estimator=None, zorder=9,
        ax=ax
    )

    sns.scatterplot(
        x=cme.cme.xs.reshape(-1), y=cme.cme.ys,
        color=color, marker="x", linewidth=1, s=20,
        zorder=10,
        ax=ax,
    )

    threshold = kmes.threshold(significance_level)

    ax.fill_between(
        x=states,
        y1=estimated_values - threshold,
        y2=estimated_values + threshold,
        color=color, alpha=0.2, linewidth=0
    )


def plot_rejection_intervals(
        intervals: list[tuple[float, float, bool]], axis_min: float, fig: plt.Figure, ax: plt.Axes
):
    for (_, x, _) in intervals[:-1]:
        ax.axvline(
            x=x,
            color=COLOR_GRID, linestyle="dotted", linewidth=0.5, zorder=1
        )

    for (x1, x2, reject) in intervals:
        highlight_patch = patches.ConnectionPatch(
            xyA=(x1, axis_min),
            xyB=(x2, axis_min),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            color="tab:green" if reject else "tab:red",
            linewidth=2,
            capstyle="butt"
        )

        fig.add_artist(highlight_patch)


def main():
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(FIGURE_WIDTH_COLUMN, FIGURE_HEIGHT),
    )

    ax_analytical, ax_bootstrap = axes

    ax_bootstrap.set_yticks([])

    def plot_run(run: Run[Config, Result], ax_):
        state_space = run.config.SPACE.discretization(run.config.resolution)

        fn_values_1 = run.result.fn_1(state_space)
        fn_values_2 = run.result.fn_2(state_space)

        plot_estimate(
            fn=run.result.fn_1, cme=run.result.cme_1, kmes=run.result.kmes_1,
            significance_level=run.config.test.significance_level,
            state_space=state_space,
            color="tab:blue",
            ax=ax_
        )

        plot_estimate(
            fn=run.result.fn_2, cme=run.result.cme_2, kmes=run.result.kmes_2,
            significance_level=run.config.test.significance_level,
            state_space=state_space,
            color="tab:orange",
            ax=ax_
        )

        axis_min = min(
            fn_values_1.min().item(),
            fn_values_2.min().item(),
        ) - 0.2

        axis_max = max(
            fn_values_1.max().item(),
            fn_values_2.max().item(),
        ) + 0.1

        ax_.set_ylim(axis_min, axis_max)

        plot_rejection_intervals(run.result.rejection_intervals(state_space), axis_min, fig, ax_)

    plot_run(run=experiment["bootstrap"], ax_=ax_bootstrap)
    plot_run(run=experiment["analytical"], ax_=ax_analytical)

    for ax in [ax_analytical, ax_bootstrap]:
        ax.margins(x=0)
        ax.grid(False)

        labels = []

        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 == 1:
                label.set_visible(False)
            else:
                num = str(label.get_text())
                if num.endswith(".0"):
                    label.set_text(num[:-2])

            labels.append(label)

        ax.set_xticklabels(labels)

        ax.tick_params(axis="both", colors="tab:gray")

    fig.legend(
        handles=[
            plt.Line2D([], [], color="tab:gray", linestyle="--"),
            plt.Line2D([], [], color="tab:gray", linestyle="-"),
            plt.Line2D([], [], color="tab:orange", linestyle="-"),
            plt.Line2D([], [], color="tab:blue", linestyle="-"),
            plt.Line2D([], [], color="tab:green", marker="s", linestyle="None"),
            plt.Line2D([], [], color="tab:red", marker="s", linestyle="None"),
        ],
        labels=[
            "Ground truth",
            "Estimate",
            "Function 1",
            "Function 2",
            "Reject",
            "Accept"
        ],
        loc="upper center",
        ncol=6,
        frameon=False,
        mode="expand",
        fontsize=TEXT_FONT_SIZE,
        handlelength=1,
        handletextpad=0.4,
        bbox_to_anchor=(0, 1.06, 1, 0),
        bbox_transform=fig.transFigure,
        handler_map={
            MulticolorPatch: MulticolorPatchHandler()
        }
    )

    ax_analytical.text(
        x=0.02, y=0.05,
        s="Analytical",
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        transform=ax_analytical.transAxes,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
        zorder=10
    )

    ax_bootstrap.text(
        x=0.02, y=0.05,
        s="Bootstrap",
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        transform=ax_bootstrap.transAxes,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
        zorder=10
    )

    plt.tight_layout(pad=0.25, w_pad=1)
    plt.subplots_adjust(top=0.875)

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIR_FIGURES / "example_1d.pdf", pad_inches=0, dpi=500)


if __name__ == "__main__":
    set_plot_style()
    main()
