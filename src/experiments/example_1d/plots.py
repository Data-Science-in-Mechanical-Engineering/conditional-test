import seaborn as sns
from jax import Array, numpy as jnp
from matplotlib import pyplot as plt, patches

from src.experiments.example_1d.spec import Config, Result
from src.rkhs import RKHSFn
from src.rkhs.testing import ConditionalTestEmbedding, ConditionedTestEmbedding


def _plot_curve(
        fn: RKHSFn, test_cme: ConditionalTestEmbedding, test_kme: ConditionedTestEmbedding, significance_level: float,
        state_space: Array, color: str, ax
):
    fn_values = fn(state_space)
    estimate_values = jnp.einsum("...i,...i->...", test_kme.kme.coefficients, test_kme.kme.points)

    sns.lineplot(x=state_space.reshape(-1), y=fn_values, color=color, linewidth=1.5, ax=ax)
    sns.lineplot(x=state_space.reshape(-1), y=estimate_values, color=color, linestyle="--", linewidth=3, ax=ax)

    threshold = test_kme.threshold(significance_level)
    tube_hi = estimate_values + threshold
    tube_lo = estimate_values - threshold

    ax.fill_between(state_space.reshape(-1), tube_lo, tube_hi, color=color, alpha=0.2, linewidth=0)

    sns.scatterplot(
        x=test_cme.cme.xs.reshape(-1), y=test_cme.cme.ys, color=color, ax=ax, marker="x", linewidth=1.5, s=80
    )


def plot_curves(config: Config, result: Result) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    state_space = config.SPACE.discretization(config.resolution)

    _plot_curve(
        result.fn_1, result.cme_1, result.kmes_1, config.test.significance_level, state_space, "tab:blue", ax
    )

    _plot_curve(
        result.fn_2, result.cme_2, result.kmes_2, config.test.significance_level, state_space, "tab:orange", ax
    )

    rejection_intervals = result.rejection_intervals(state_space)
    axis_min = ax.get_ylim()[0]

    for (_, x, _) in rejection_intervals[:-1]:
        ax.axvline(x=x, color="tab:gray", linestyle="dotted", linewidth=0.5, zorder=10)

    for (x1, x2, reject) in rejection_intervals:
        decision_patch = patches.ConnectionPatch(
            xyA=(x1, axis_min), xyB=(x2, axis_min),
            coordsA="data", coordsB="data",
            axesA=ax, axesB=ax,
            color="tab:red" if reject else "tab:green", linewidth=1, capstyle="butt"
        )

        fig.add_artist(decision_patch)

    ax.margins(x=0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    return fig
