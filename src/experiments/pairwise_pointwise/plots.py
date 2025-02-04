import math
from typing import Callable, Any

import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import AnalyticalThresholdConfig, BootstrapThresholdConfig, SpaceConfig
from src.data import RKHSFnSampling
from src.experiments.pairwise_pointwise.spec import Config, Result, MeanConfig, DistributionalConfig
from src.figures.rkhs import plot_rkhs_fn_2d, plot_rkhs_fn_sampling_2d
from src.rkhs import RKHSFn, Kernel


def plot_state_space_heatmap(grid: jnp.ndarray, max_scale: float, ax):
    assert grid.ndim == 2

    sns.heatmap(
        grid, ax=ax,
        vmin=0, vmax=max_scale,
        xticklabels=False, yticklabels=False,
        cmap="inferno", cbar=False
    )


def plot_grid(
        functions: RKHSFn, kernel_x: Kernel,
        space: SpaceConfig, granularity: int,
        plot_cell_fn: Callable[[int, int, plt.Axes], Any],
        max_scale: float,
        title: str
):
    n_functions = functions.coefficients.shape[0]

    fig = plt.figure(figsize=(3 * n_functions, 3 * n_functions))

    grid = GridSpec(
        nrows=n_functions + 1,
        ncols=n_functions + 2,
        width_ratios=[0.5] + [1] * n_functions + [0.25],
        height_ratios=[0.5] + [1] * n_functions,
        figure=fig
    )

    for i in range(1, n_functions + 1):
        ax_left = fig.add_subplot(grid[i, 0], projection="3d")
        ax_top = fig.add_subplot(grid[0, i], projection="3d")

        plot_rkhs_fn_2d(kernel_x, functions[i - 1], space, granularity, ax=ax_left, show_grid=False)
        plot_rkhs_fn_2d(kernel_x, functions[i - 1], space, granularity, ax=ax_top, show_grid=False)

    for i in range(n_functions):
        for j in range(n_functions):
            ax = fig.add_subplot(grid[i + 1, j + 1])

            plot_cell_fn(i, j, ax)
            ax.set_xticks([])
            ax.set_yticks([])

    cbar_ax = fig.add_subplot(grid[1:, -1])

    norm = plt.Normalize(vmin=0, vmax=max_scale)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cbar_ax)

    colorbar.outline.set_visible(False)

    fig.suptitle(title)

    plt.tight_layout()

    return fig


def plot_data_grid(
        functions: RKHSFn, kernel_x: Kernel,
        space: SpaceConfig, resolution: int,
        datasets: RKHSFnSampling,
        title: str
):
    n_functions = functions.coefficients.shape[0]
    n_datasets = datasets.ys.shape[0]
    assert n_datasets == n_functions

    n_rows = math.ceil(math.sqrt(n_functions))
    n_cols = n_functions // n_rows

    fig = plt.figure(figsize=(3 * n_functions, 3 * n_functions))

    grid = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)

    ax = None

    for i in range(n_datasets):
        fn, dataset = functions[i], datasets[i]

        ax = fig.add_subplot(
            grid[i // n_cols, i % n_cols], projection="3d", computed_zorder=False,
            sharex=ax, sharey=ax, sharez=ax
        )

        plot_rkhs_fn_sampling_2d(
            kernel_x, fn, dataset,
            space, resolution,
            ax=ax,
            show_grid=False
        )

        ax.view_init(elev=70)

    fig.suptitle(title)

    plt.tight_layout()

    return fig


def plot_data(config: Config, result: Result) -> dict[str, plt.Figure]:
    kernel = config.parametrization().make().x

    return {
        "datasets_1": plot_data_grid(
            functions=result.functions,
            kernel_x=kernel,
            space=config.data.space,
            resolution=config.resolution,
            datasets=result.datasets_1,
            title="Dataset 1"
        ),
        "datasets_2": plot_data_grid(
            functions=result.functions,
            kernel_x=kernel,
            space=config.data.space,
            resolution=config.resolution,
            datasets=result.datasets_2,
            title="Dataset 2"
        )
    }


def plot_thresholds(config: MeanConfig | DistributionalConfig, result: Result) -> plt.Figure:
    threshold = config.threshold

    if isinstance(threshold, AnalyticalThresholdConfig):
        threshold_type = "analytical"
    elif isinstance(threshold, BootstrapThresholdConfig):
        threshold_type = rf"bootstrapped, N={threshold.n_bootstrap}"
    else:
        threshold_type = "?"

    return plot_grid(
        functions=result.functions, kernel_x=config.parametrization().make().x,
        space=config.data.space, granularity=config.resolution,
        plot_cell_fn=lambda i, j, ax: plot_state_space_heatmap(
            result.threshold_grid_pairwise[i, j],
            result.cmmd_max_scale,
            ax
        ),
        max_scale=result.cmmd_max_scale,
        title=rf"Thresholds for $\alpha={threshold.confidence_level}$ ({threshold_type})"
    )


def plot_rejection_region(config: MeanConfig | DistributionalConfig, result: Result) -> plt.Figure:
    threshold = config.threshold

    if isinstance(threshold, AnalyticalThresholdConfig):
        threshold_type = "analytical"
    elif isinstance(threshold, BootstrapThresholdConfig):
        threshold_type = rf"bootstrapped, N={threshold.n_bootstrap}"
    else:
        threshold_type = "?"

    return plot_grid(
        functions=result.functions, kernel_x=config.parametrization().make().x,
        space=config.data.space, granularity=config.resolution,
        plot_cell_fn=lambda i, j, ax: plot_state_space_heatmap(result.rejection_grid_pairwise[i, j], 1, ax),
        max_scale=1,
        title=rf"Rejection region for $\alpha={threshold.confidence_level}$ ({threshold_type})"
    )


def plot_cmmd_estimate(config: Config, result: Result) -> plt.Figure:
    return plot_grid(
        functions=result.functions, kernel_x=config.parametrization().make().x,
        space=config.data.space, granularity=config.resolution,
        plot_cell_fn=lambda i, j, ax: plot_state_space_heatmap(
            result.cmmd_grid_pairwise[i, j],
            result.cmmd_max_scale,
            ax
        ),
        max_scale=result.cmmd_max_scale,
        title="Estimated CMMD between independent datasets of different RKHS functions"
    )


def plot_true_cmmd(config: Config, result: Result) -> plt.Figure:
    return plot_grid(
        functions=result.functions, kernel_x=config.parametrization().make().x,
        space=config.data.space, granularity=config.resolution,
        plot_cell_fn=lambda i, j, ax: plot_state_space_heatmap(
            result.true_cmmd_grid_pairwise[i, j],
            result.cmmd_max_scale,
            ax
        ),
        max_scale=result.cmmd_max_scale,
        title="Ground truth MMD between RKHS function pairs"
    )
