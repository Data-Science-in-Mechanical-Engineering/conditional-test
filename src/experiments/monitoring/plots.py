from typing import Callable
from typing import Literal

import seaborn as sns
from jax import Array, numpy as jnp
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from src.experiments.monitoring.spec import SingleResult, Config, MultipleResult


def _plot_statistic_ci(
        runs: list[SingleResult], statistic: Callable[[SingleResult], Array],
        significance_level: float, t_change: int, t_adapted: int, label: str,
        ax: plt.Axes
):
    statistics = jnp.stack([statistic(run) for run in runs])

    mean_statistic = statistics.mean(axis=0)
    lo_quantile_statistic = jnp.quantile(statistics, q=significance_level, axis=0)
    hi_quantile_statistic = jnp.quantile(statistics, q=1 - significance_level, axis=0)

    assert mean_statistic.ndim == 1
    assert lo_quantile_statistic.ndim == 1
    assert hi_quantile_statistic.ndim == 1
    assert mean_statistic.shape == lo_quantile_statistic.shape
    assert mean_statistic.shape == hi_quantile_statistic.shape

    color = sns.color_palette("bright")[0]

    time = jnp.arange(len(mean_statistic))
    sns.lineplot(x=time, y=mean_statistic, ax=ax, color=color, linewidth=2)

    ax.fill_between(time, lo_quantile_statistic, hi_quantile_statistic, alpha=0.1, color=color)

    ax.axvline(x=t_change, color="black", linestyle="dotted", linewidth=0.5)
    ax.axvline(x=t_adapted, color="black", linestyle="dotted", linewidth=0.5)

    ax.set_xlabel("Time step")
    ax.set_ylabel(label)

    ax.set_xmargin(0)
    ax.grid(False)

    plt.tight_layout()


def plot_beta(config: Config, result: MultipleResult) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def statistic(run: SingleResult) -> Array:
        return run.beta

    _plot_statistic_ci(
        runs=result.runs,
        statistic=statistic,
        significance_level=config.test.significance_level,
        t_change=config.t_change,
        t_adapted=config.t_adapted,
        label=r"Mean $\beta$",
        ax=ax
    )

    return fig


def _plot_posterior_std(config: Config, result: MultipleResult, dataset: Literal["reference", "window"]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def statistic(run: SingleResult) -> Array:
        if dataset == "reference":
            return run.posterior_std_reference.mean(axis=-1)
        elif dataset == "window":
            return run.posterior_std_windows.mean(axis=-1)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    _plot_statistic_ci(
        runs=result.runs,
        statistic=statistic,
        significance_level=config.test.significance_level,
        t_change=config.t_change,
        t_adapted=config.t_adapted,
        label=rf"Mean $\sigma(x)$",
        ax=ax
    )

    return fig


def plot_posterior_std(config: Config, result: MultipleResult) -> dict[str, plt.Figure]:
    return {
        "reference": _plot_posterior_std(config, result, "reference"),
        "window": _plot_posterior_std(config, result, "window"),
    }


def plot_thresholds(config: Config, result: MultipleResult) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def statistic(run: SingleResult) -> Array:
        return run.outcomes.threshold.mean(axis=-1)

    _plot_statistic_ci(
        runs=result.runs,
        statistic=statistic,
        significance_level=config.test.significance_level,
        t_change=config.t_change,
        t_adapted=config.t_adapted,
        label="Mean threshold",
        ax=ax
    )

    return fig


def plot_cmmd(config: Config, result: MultipleResult) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def statistic(run: SingleResult) -> Array:
        return run.outcomes.distance.mean(axis=-1)

    _plot_statistic_ci(
        runs=result.runs,
        statistic=statistic,
        significance_level=config.test.significance_level,
        t_change=config.t_change,
        t_adapted=config.t_adapted,
        label="Mean CMMD",
        ax=ax
    )

    return fig


def plot_ratio(config: Config, result: MultipleResult) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def statistic(run: SingleResult) -> Array:
        ratio = run.outcomes.distance / run.outcomes.threshold
        return ratio.mean(axis=-1)

    _plot_statistic_ci(
        runs=result.runs,
        statistic=statistic,
        significance_level=config.test.significance_level,
        t_change=config.t_change,
        t_adapted=config.t_adapted,
        label="Max. ratio",
        ax=ax
    )

    ax.axhline(y=1, color="tab:gray", linestyle="dotted", linewidth=0.5)

    return fig


def _tsne_trajectory(data: Array) -> Array:
    assert data.ndim == 2

    tsne = TSNE(n_components=2, random_state=0)
    return tsne.fit_transform(data)


def _plot_online_trajectory(trajectory: Array, length_nominal: int) -> plt.Figure:
    assert trajectory.ndim == 2

    dim = trajectory.shape[1]

    if dim > 2:
        trajectory = _tsne_trajectory(trajectory)

    assert trajectory.ndim == 2

    nominal_trajectory = trajectory[:length_nominal]
    anomalous_trajectory = trajectory[length_nominal:]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def plot(data: Array, label: str, color: str):
        sns.lineplot(
            x=data[:, 0], y=data[:, 1],
            color=color, linewidth=0.5, alpha=0.5,
            label=label, sort=False, ax=ax
        )

        sns.scatterplot(
            x=data[:, 0], y=data[:, 1],
            color=color,
            ax=ax
        )

    plot(nominal_trajectory, label="nominal", color="tab:green")
    plot(anomalous_trajectory, label="disturbed", color="tab:red")

    ax.grid(False)

    if dim == 2:
        ax.set_title("Online trajectory")
    else:
        ax.set_title("t-SNE of the online trajectory")

    return fig


def plot_online_trajectories(config: Config, result: MultipleResult) -> dict[str, plt.Figure]:
    return {
        f"run-{i + 1:03d}": _plot_online_trajectory(
            trajectory=run.online_trajectory,
            length_nominal=config.length_nominal_trajectory
        )
        for i, run in enumerate(result.runs[:1])
    }
