import jax.numpy as jnp
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib import pyplot as plt, patches
from matplotlib.gridspec import GridSpec

from src.experiments.example_1d.spec import Result, Config
from src.figures.util import TEXT_FONT_SIZE


def plot_estimate(
        states: jnp.ndarray,
        values: jnp.ndarray, estimated_values: jnp.ndarray,
        dataset_xs: jnp.ndarray, dataset_ys: jnp.ndarray,
        beta: float, sigmas: jnp.ndarray,
        ax,
        color
):
    sns.lineplot(x=states, y=values, color=color, linewidth=1, ax=ax)
    sns.lineplot(x=states, y=estimated_values, color=color, linewidth=2, linestyle="--", ax=ax)

    sns.scatterplot(
        x=dataset_xs.reshape(-1), y=dataset_ys,
        color=color,
        marker="x", s=20, linewidth=1,
        ax=ax
    )

    ax.fill_between(
        x=states,
        y1=estimated_values - beta * sigmas,
        y2=estimated_values + beta * sigmas,
        color=color, alpha=0.2, linewidth=0
    )


def plot_rejection_intervals(rejection_intervals: list[tuple[float, float, bool]], axis_min: float, fig, ax):
    for (_, x, _) in rejection_intervals[:-1]:
        ax.axvline(
            x=x,
            color="tab:gray", linestyle="dotted", linewidth=0.5, zorder=10
        )

    for (x1, x2, reject) in rejection_intervals:
        highlight_patch = patches.ConnectionPatch(
            xyA=(x1, axis_min),
            xyB=(x2, axis_min),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            color="tab:red" if reject else "tab:green",
            linewidth=1,
            capstyle="butt"
        )

        fig.add_artist(highlight_patch)


def plot_legend(ax_legend):
    ax_legend.legend(
        handles=[
            mlines.Line2D([], [], color="tab:gray", linestyle="-", linewidth=1),
            mlines.Line2D([], [], color="tab:gray", linestyle="--", linewidth=1),
            # mlines.Line2D([], [], color="tab:gray", marker="x", linestyle="None", linewidth=1.5, markersize=5),
            mlines.Line2D([], [], color="tab:red", marker="s", linestyle="None"),
            mlines.Line2D([], [], color="tab:green", marker="s", linestyle="None"),
        ],
        labels=[
            "Ground truth",
            "Estimate",
            # "Sample",
            "Reject",
            "Accept"
        ],
        loc="upper left",
        ncol=5,
        facecolor="white",
        mode="expand",
        frameon=False,
        bbox_to_anchor=(0, 1.25, 1, 0),
        fontsize=TEXT_FONT_SIZE,
        handlelength=1,  # Adjusts the length of the legend lines
        handletextpad=0.25,  # Controls the padding between the symbol and text
    )


def _plot_outcome(config: Config, result: Result, analytical: bool) -> plt.Figure:
    if analytical:
        beta_1, beta_2 = result.beta_1_analytical, result.beta_2_analytical
        thresholds = result.thresholds_analytical()
    else:
        beta_1, beta_2 = result.beta_1_botstrap, result.beta_2_bootstrap
        thresholds = result.thresholds_bootstrap()

    state_space = config.state_space()
    flat_states = config.state_space().reshape(-1)

    fig = plt.figure(figsize=(8, 5))
    grid_spec = GridSpec(nrows=2, ncols=1, height_ratios=[0.05, 0.95])

    ax_legend = fig.add_subplot(grid_spec[0])
    ax_main = fig.add_subplot(grid_spec[1])

    ax_legend.axis("off")
    plot_legend(ax_legend)

    plot_estimate(
        states=flat_states,
        values=result.values_1, estimated_values=result.estimated_values_1,
        dataset_xs=result.dataset_1.xs, dataset_ys=result.dataset_1.ys,
        beta=float(beta_1), sigmas=result.sigmas_1,
        ax=ax_main,
        color="tab:orange"
    )

    plot_estimate(
        states=flat_states,
        values=result.values_2, estimated_values=result.estimated_values_2,
        dataset_xs=result.dataset_2.xs, dataset_ys=result.dataset_2.ys,
        beta=float(beta_2), sigmas=result.sigmas_2,
        ax=ax_main,
        color="tab:blue"
    )

    axis_min = min(float(result.values_1.min()), float(result.values_2.min())) - 0.2
    axis_max = max(float(result.values_1.max()), float(result.values_2.max())) + 0.2
    ax_main.set_ylim(axis_min, axis_max)

    plot_rejection_intervals(
        rejection_intervals=result.rejection_intervals(state_space, thresholds),
        axis_min=axis_min,
        fig=fig,
        ax=ax_main
    )

    ax_main.margins(x=0)

    ax_main.grid(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    plt.tight_layout(pad=1)

    return fig


def plot_analytical(config: Config, result: Result) -> plt.Figure:
    return _plot_outcome(config, result, analytical=True)


def plot_bootstrap(config: Config, result: Result) -> plt.Figure:
    return _plot_outcome(config, result, analytical=False)
