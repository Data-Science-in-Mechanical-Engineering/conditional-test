import seaborn as sns

from src.config import SpaceConfig
from src.data import RKHSFnSampling
from src.rkhs import Kernel, RKHSFn


def plot_rkhs_fn_1d(
        kernel: Kernel, fn: RKHSFn,
        space: SpaceConfig, resolution: int,
        ax,
        color=None
):
    xs = space.discretization(resolution)
    ys = kernel.evaluate.one_many(fn, xs)

    sns.lineplot(x=xs.reshape(-1), y=ys, ax=ax, color=color)


def plot_rkhs_fn_2d(
        kernel: Kernel, fn: RKHSFn,
        space: SpaceConfig, resolution: int,
        ax,
        show_grid: bool = False
):
    assert space.dim == 2

    flat_grid = space.discretization(resolution)
    grid_x = flat_grid[:, 0].reshape(resolution, resolution)
    grid_y = flat_grid[:, 1].reshape(resolution, resolution)

    function_value_grid = kernel.evaluate.one_many(fn, flat_grid).reshape(resolution, resolution)

    ax.plot_surface(grid_x, grid_y, function_value_grid, cmap="viridis", alpha=0.8, zorder=1)

    if not show_grid:
        ax.set_axis_off()


def plot_rkhs_fn_sampling_2d(
        kernel: Kernel, fn: RKHSFn,
        sampling: RKHSFnSampling,
        space: SpaceConfig, resolution: int,
        ax,
        show_grid: bool = False
):
    plot_rkhs_fn_2d(kernel, fn, space, resolution, ax, show_grid=show_grid)

    assert sampling.xs.ndim in {2, 3}
    assert sampling.xs.shape[-1] == 2

    if sampling.xs.ndim == 2:
        xs = sampling.xs[None, ...]
        ys = sampling.ys[None, ...]
    else:
        xs = sampling.xs
        ys = sampling.ys

    for xs, ys in zip(xs, ys):
        ax.scatter(xs[:, 0], xs[:, 1], ys, c=ys, cmap="coolwarm", alpha=0.5)
