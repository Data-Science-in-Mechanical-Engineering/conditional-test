import seaborn as sns
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from src.experiments.error_rates.spec import PositiveRate, BaseConfig


def plot_positive_rates(config: BaseConfig, rates: PositiveRate) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5))
    grid = GridSpec(figure=fig, nrows=1, ncols=2)

    ax_uniform = fig.add_subplot(grid[0, 0])
    ax_local = fig.add_subplot(grid[0, 1], sharex=ax_uniform, sharey=ax_uniform)

    ax_uniform.set_xlabel(r"$\alpha$")
    ax_local.set_xlabel(r"$\alpha$")

    ax_uniform.set_title("Space-uniform positive rate")
    ax_local.set_title("Space-local positive rate")

    confidence_levels = config.confidence_levels()

    sns.lineplot(
        x=confidence_levels, y=rates.uniform,
        ax=ax_uniform,
        linewidth=3,
        estimator=None
    )

    sns.lineplot(
        x=confidence_levels, y=rates.local,
        ax=ax_local,
        linewidth=3,
        estimator=None
    )

    ax_uniform.set_ylim(-0.025, 1.025)

    plt.tight_layout(pad=1)

    return fig
