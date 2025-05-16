import seaborn as sns
from jax import Array, numpy as jnp
from matplotlib import pyplot as plt

from src.experiments.error_rates.spec import Config
from src.util import COLOR_GRID


def plot_positive_rate(config: Config, positive_rates: Array) -> plt.Figure:
    significance_levels = config.significance_levels

    assert significance_levels.ndim == 1
    assert positive_rates.ndim == 2
    assert len(significance_levels) == positive_rates.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mean = positive_rates.mean(axis=0)

    sns.lineplot(
        x=significance_levels, y=mean,
        linewidth=2,
        ax=ax
    )

    identity = jnp.linspace(0, 1, 1000)

    sns.lineplot(
        x=identity, y=identity,
        color=COLOR_GRID, linestyle="-", linewidth=0.5,
        ax=ax
    )

    ax.set_xmargin(0.02)
    ax.set_ymargin(0.02)

    ax.set_xlabel(r"Significance level $\alpha$")
    fig.suptitle("Positive rate")

    plt.tight_layout()

    return fig
