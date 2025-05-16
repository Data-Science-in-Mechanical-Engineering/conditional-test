from typing import NamedTuple, Self, Mapping, Iterable, Hashable, Literal

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import tyro
from jax import Array
from typing_extensions import Callable

from src.experiments.error_rates.__main__ import OutputKernelExperiment, MixtureNoiseExperiment, DatasetSizeExperiment, \
    BootstrapVsAnalyticalExperiment, LocalDisturbanceExperiment, GlobalDisturbanceExperiment
from src.experiments.error_rates.spec import Config
from src.expyro.experiment import Run
from src.figures.util import make_color_palette, make_marker_palette
from src.util import FIGURE_WIDTH_COLUMN, set_plot_style, TEXT_FONT_SIZE, DIR_FIGURES, COLOR_GRID


def _extract_shared_parameter[K, T: Hashable](collection: Iterable[K], key: Callable[[K], T]) -> T:
    parameters = set()

    for element in collection:
        parameter = key(element)
        parameters.add(parameter)

    if not parameters:
        raise ValueError("No parameter found.")

    if len(parameters) > 1:
        print(parameters)
        raise ValueError("Multiple parameters found.")

    return parameters.pop()


def extract_significance_levels(runs: Iterable[Run[Config, Array]]) -> Array:
    significance_levels = _extract_shared_parameter(runs, lambda run: tuple(map(float, run.config.significance_levels)))
    return jnp.array(significance_levels)


class LineStyle(NamedTuple):
    style: str
    color: str
    label: str
    width: float
    marker: str | None

    def plot(self, x: Array, y: Array, ax: plt.Axes, z: int) -> None:
        sns.lineplot(
            x=x, y=y,
            linewidth=self.width, linestyle=self.style, color=self.color, label=self.label, zorder=z,
            legend=False,
            ax=ax
        )


class ErrorRateFigure(NamedTuple):
    fig: plt.Figure
    axes: list[plt.Axes]
    n_legends: int
    height: float

    def tighten(self):
        self.fig.tight_layout(pad=0.25)
        self.fig.subplots_adjust(top=0.95 - 0.1 * self.n_legends * 1.5 / self.height)

    @classmethod
    def default(cls, n_legends: int, n_plots: int, width: float = FIGURE_WIDTH_COLUMN, height: float = 1.5) -> Self:
        fig, axes = plt.subplots(
            ncols=n_plots, figsize=(width, height + n_legends * 0.2),
            sharey=True, sharex=True,
        )

        for ax in ([axes] if n_plots == 1 else axes):
            ax.set_xlabel(r"$\alpha$")

            ax.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
            ax.minorticks_on()

            ax.set_xmargin(0)
            ax.set_ylim(-0.025, 1.025)
            ax.set_xlim(0, 1)

        return ErrorRateFigure(
            fig=fig,
            axes=[axes] if n_plots == 1 else [axes[i] for i in range(n_plots)],
            n_legends=n_legends,
            height=height
        )

    def add_rates[K: Hashable](
            self,
            i: int,
            run_batches: Mapping[K, list[Run[Config, Array]]],
            metric: Callable[[Array], Array],
            line_fn: Callable[[K], LineStyle],
            n_markers: int | None = None,
            initial_marker_offset: int = 0
    ):
        ax = self.axes[i]

        if n_markers is None:
            n_markers = 4

        n_batches = len(run_batches)
        marker_offset = 0.03
        marker_positions = jnp.linspace(0, 1 - n_batches * marker_offset, n_markers + 2)[1:-1]
        marker_positions = marker_positions - initial_marker_offset * marker_offset

        for i, (key, runs) in enumerate(run_batches.items()):
            significance_levels = extract_significance_levels(runs)
            rejections = jnp.stack([run.result for run in runs], axis=0)

            assert rejections.ndim == 3
            assert rejections.shape[1] == len(significance_levels)

            rejection_rates = rejections.mean(axis=1)
            metrics = metric(rejection_rates)

            mean_metric = metrics.mean(axis=0)
            lo_ci_rate = jnp.quantile(metrics, 0.025, axis=0)
            hi_ci_rate = jnp.quantile(metrics, 0.975, axis=0)

            line = line_fn(key)
            line.plot(x=significance_levels, y=mean_metric, ax=ax, z=10 + i)

            ax.fill_between(
                x=significance_levels, y1=lo_ci_rate, y2=hi_ci_rate,
                alpha=0.1, color=line.color, linewidth=0,
            )

            marker_indices = jnp.round(len(significance_levels) * (marker_positions + i * marker_offset)).astype(int)

            if line.marker is not None:
                ax.scatter(
                    x=significance_levels[marker_indices], y=mean_metric[marker_indices],
                    marker=line.marker, color=line.color, label=line.label, s=12, zorder=10 + i,
                )

    def add_legends(self, legends: list[list[LineStyle | None]]):
        assert len(legends) >= 1
        n_cols = len(legends[0])
        assert all(len(styles) == n_cols for styles in legends)

        styles = [
            style
            for combinations in zip(*legends)
            for style in combinations
        ]

        handles = [
            plt.Line2D(
                [], [],
                linewidth=1.5,
                linestyle=style.style,
                color=style.color,
                marker=style.marker,
                markersize=4
            )
            if style is not None else
            plt.Line2D([], [], color="white", linestyle="-", linewidth=0)
            for style in styles
        ]

        labels = [
            style.label if style is not None else ""
            for style in styles
        ]

        self.fig.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            ncol=n_cols,
            frameon=False,
            mode="expand",
            bbox_to_anchor=(0, 1.05, 1, 0),
            fontsize=TEXT_FONT_SIZE,
            handlelength=1.5,
            handletextpad=0.5,
        )

    def add_identity(self, i: int, sign: Literal[+1, -1] = +1):
        ax = self.axes[i]

        identity = jnp.linspace(0, 1, 101).at[0].set(0.001)

        sns.lineplot(
            x=identity, y=sign * identity - (sign - 1) // 2,
            color=COLOR_GRID, linestyle="-", linewidth=0.5,
            legend=False,
            ax=ax
        )

    def __getitem__(self, item: int) -> plt.Axes:
        return self.axes[item]

    def __len__(self):
        return len(self.axes)

    def __iter__(self):
        return iter(self.axes)


def gaussian_vs_polynomial():
    figure = ErrorRateFigure.default(n_legends=2, n_plots=2)
    figure[0].set_ylabel("Type I Error")
    figure[1].set_ylabel("Type II Error")

    degrees = [1, 2, 3]
    bandwidths = [0.05, 0.1, 0.15]

    colors = make_color_palette(
        palette="bright",
        values=[("gaussian", bandwidth) for bandwidth in bandwidths] + [("polynomial", degree) for degree in degrees]
    )

    markers = make_marker_palette(["gaussian", "polynomial"])

    h0_runs_polynomial = {
        degree: OutputKernelExperiment.load_runs(f"polynomial/H0/parameter={degree}")
        for degree in degrees
    }

    h1_runs_polynomial = {
        degree: OutputKernelExperiment.load_runs(f"polynomial/H1/parameter={degree}")
        for degree in degrees
    }

    h0_runs_gaussian = {
        bandwidth: OutputKernelExperiment.load_runs(f"gaussian/H0/parameter={bandwidth}")
        for bandwidth in bandwidths
    }

    h1_runs_gaussian = {
        bandwidth: OutputKernelExperiment.load_runs(f"gaussian/H1/parameter={bandwidth}")
        for bandwidth in bandwidths
    }

    def style_gaussian(bandwidth: float) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors["gaussian", bandwidth],
            width=1.5,
            marker=markers["gaussian"],
            label=fr"$\gamma^2={bandwidth:.2f}$"
        )

    def style_polynomial(degree: int) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors["polynomial", degree],
            width=1.5,
            marker=markers["polynomial"],
            label=fr"$d={degree}$"
        )

    figure.add_rates(
        i=0,
        run_batches=h0_runs_gaussian,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_gaussian,
    )

    figure.add_rates(
        i=0,
        run_batches=h0_runs_polynomial,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_polynomial,
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_gaussian,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_gaussian,
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_polynomial,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_polynomial,
    )

    figure.add_identity(i=0, sign=+1)
    figure.add_identity(i=1, sign=-1)

    figure.add_legends([
        [
            LineStyle(style="-", color="tab:gray", label="Gaussian", width=1.5, marker=markers["gaussian"]),
            None,
            None
        ] + [
            style_gaussian(bandwidth)
            for bandwidth in sorted(bandwidths)
        ],
        [
            LineStyle(style="-", color="tab:gray", label="Polynomial", width=1.5, marker=markers["polynomial"]),
            None,
            None
        ] + [
            style_polynomial(degree)
            for degree in sorted(degrees)
        ]
    ])

    figure.tighten()
    DIR_FIGURES.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURES / "gaussian_vs_polynomial.pdf", dpi=500)


def gaussian_vs_linear():
    figure = ErrorRateFigure.default(n_legends=1, n_plots=1)
    figure[0].set_ylabel("Positive rate")

    noise_means = [0.05, 0.075, 0.1]
    colors = make_color_palette("bright", noise_means)
    markers = make_marker_palette(["gaussian", "linear"])

    runs_gaussian = {
        noise_mean: MixtureNoiseExperiment.load_runs(f"gaussian/H1/noise_mean={noise_mean}")
        for noise_mean in noise_means
    }

    runs_linear = {
        noise_mean: MixtureNoiseExperiment.load_runs(f"linear/H1/noise_mean={noise_mean}")
        for noise_mean in noise_means
    }

    def style_gaussian(noise_mean: float) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors[noise_mean],
            width=1.5,
            marker=markers["gaussian"],
            label=fr"$\sigma={noise_mean:.2f}$"
        )

    def style_linear(noise_mean: float) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors[noise_mean],
            width=1.5,
            marker=markers["linear"],
            label=fr"$\sigma={noise_mean:.2f}$"
        )

    figure.add_rates(
        i=0,
        run_batches=runs_gaussian,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_gaussian
    )

    figure.add_rates(
        i=0,
        run_batches=runs_linear,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_linear
    )

    figure.add_identity(i=0, sign=+1)

    figure.add_legends([
        [
            LineStyle(style="-", color="tab:gray", label="Gaussian", width=1.5, marker=markers["gaussian"]),
            LineStyle(style="-", color="tab:gray", label="Linear", width=1.5, marker=markers["linear"]),
            None,
        ] + [
            LineStyle(
                style="-",
                color=colors[noise_mean],
                width=1.5,
                marker=None,
                label=fr"$\mu={noise_mean:.3f}$"
            )
            for noise_mean in sorted(noise_means)
        ]
    ])

    figure.tighten()
    DIR_FIGURES.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURES / "gaussian_vs_linear.pdf", dpi=500)


def dataset_size():
    figure = ErrorRateFigure.default(n_legends=1, n_plots=2)
    figure[0].set_ylabel("Type I Error")
    figure[1].set_ylabel("Type II Error")

    sizes = [20, 50, 100, 250]
    colors = make_color_palette("bright", sizes)
    markers = make_marker_palette(sizes)

    h0_runs = {
        size: DatasetSizeExperiment.load_runs(f"bootstrap/H0/size={size}")
        for size in sizes
    }

    h1_runs = {
        size: DatasetSizeExperiment.load_runs(f"bootstrap/H1/size={size}")
        for size in sizes
    }

    def style(size: int) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors[size],
            width=1.5,
            marker=markers[size],
            label=fr"$N={size}$"
        )

    figure.add_rates(
        i=0,
        run_batches=h0_runs,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style
    )

    figure.add_identity(i=0, sign=+1)
    figure.add_identity(i=1, sign=-1)

    figure.add_legends([
        [style(size) for size in sorted(sizes)]
    ])

    figure.tighten()
    DIR_FIGURES.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURES / "dataset_size.pdf", dpi=500)


def bootstrap_vs_analytical():
    figure = ErrorRateFigure.default(n_legends=1, n_plots=2)
    figure[0].set_ylabel("Type I Error")
    figure[1].set_ylabel("Type II Error")

    noise_stds = [0.0, 0.01, 0.1, 0.5]

    colors = make_color_palette("bright", noise_stds)
    markers = make_marker_palette(["analytical", "bootstrap"])

    h0_runs_analytical = {
        noise_std: BootstrapVsAnalyticalExperiment.load_runs(f"analytical/H0/noise_std={noise_std}")
        for noise_std in noise_stds
    }

    h1_runs_analytical = {
        noise_std: BootstrapVsAnalyticalExperiment.load_runs(f"analytical/H1/noise_std={noise_std}")
        for noise_std in noise_stds
    }

    h0_runs_bootstrap = {
        noise_std: BootstrapVsAnalyticalExperiment.load_runs(f"bootstrap/H0/noise_std={noise_std}")
        for noise_std in noise_stds
    }

    h1_runs_bootstrap = {
        noise_std: BootstrapVsAnalyticalExperiment.load_runs(f"bootstrap/H1/noise_std={noise_std}")
        for noise_std in noise_stds
    }

    def style_analytical(noise_std: float) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors[noise_std],
            width=1.5,
            marker=markers["analytical"],
            label=fr"$\sigma={noise_std:.2f}$"
        )

    def style_bootstrap(noise_std: float) -> LineStyle:
        return LineStyle(
            style="-",
            color=colors[noise_std],
            width=1.5,
            marker=markers["bootstrap"],
            label=fr"$\sigma={noise_std:.2f}$"
        )

    figure.add_rates(
        i=0,
        run_batches=h0_runs_analytical,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_analytical
    )

    figure.add_rates(
        i=0,
        run_batches=h0_runs_bootstrap,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_bootstrap
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_analytical,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_analytical
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_bootstrap,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_bootstrap
    )

    figure.add_identity(i=0, sign=+1)
    figure.add_identity(i=1, sign=-1)

    figure.add_legends([
        [
            LineStyle(style="-", color="tab:gray", label="Analytical", width=1.5, marker=markers["analytical"]),
            LineStyle(style="-", color="tab:gray", label="Bootstrap", width=1.5, marker=markers["bootstrap"]),
        ] + [
            LineStyle(
                style="-",
                color=colors[noise_std],
                label=fr"$\sigma={noise_std:.2f}$",
                width=1.5,
                marker=None
            )
            for noise_std in noise_stds
        ]
    ])

    figure.tighten()
    DIR_FIGURES.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURES / "bootstrap_vs_analytical.pdf", dpi=500)


def our_vs_baseline():
    figure = ErrorRateFigure.default(n_legends=2, n_plots=3, height=1.25)
    figure[0].set_ylabel("Type I Error")
    figure[1].set_ylabel("Type II Error")
    figure[2].set_ylabel("Type II Error")

    norms = [0.1, 0.25, 0.5]
    weights = [0.01, 0.02, 0.04]
    tolerance = 0.01
    disturbance_norm = 1.0

    colors = make_color_palette(
        palette="bright",
        values=[("global", norm) for norm in norms] + [("local", weight) for weight in weights]
    )

    markers = make_marker_palette(["ours", "hu-lei"])

    line_style = {
        "ours": "-",
        "hu-lei": ":"
    }

    h0_runs = {
        "ours": GlobalDisturbanceExperiment.load_runs("bootstrap/H0"),
        "hu-lei": GlobalDisturbanceExperiment.load_runs("hu-lei__hg/H0"),
    }

    h1_runs_global_ours = {
        norm: GlobalDisturbanceExperiment.load_runs(f"bootstrap/H1/norm={norm}")
        for norm in norms
    }

    h1_runs_global_hu_lei = {
        norm: GlobalDisturbanceExperiment.load_runs(f"hu-lei__hg/H1/norm={norm}")
        for norm in norms
    }

    h1_runs_local_ours = {
        weight: LocalDisturbanceExperiment.load_runs(f"bootstrap/H1/weight={weight}__tol={tolerance}"
                                                     f"__norm={disturbance_norm}")
        for weight in weights
    }

    h1_runs_hu_lei_local = {
        weight: LocalDisturbanceExperiment.load_runs(f"hu-lei__gt/H1/weight={weight}__tol={tolerance}"
                                                     f"__norm={disturbance_norm}")
        for weight in weights
    }

    def style_h0(name: str) -> LineStyle:
        return LineStyle(
            style=line_style[name],
            color="black",
            label="",
            width=1.5,
            marker=markers[name],
        )

    def style_ours_global(norm: float) -> LineStyle:
        return LineStyle(
            style=line_style["ours"],
            color=colors["global", norm],
            width=1.5,
            marker=markers["ours"],
            label=fr"$\xi={norm:.2f}$"
        )

    def style_hu_lei_global(norm: float) -> LineStyle:
        return LineStyle(
            style=line_style["hu-lei"],
            color=colors["global", norm],
            width=1.5,
            marker=markers["hu-lei"],
            label=fr"$\xi={norm:.2f}$"
        )

    def style_ours_local(disturbance: float) -> LineStyle:
        return LineStyle(
            style=line_style["ours"],
            color=colors["local", disturbance],
            width=1.5,
            marker=markers["ours"],
            label=fr"$\theta={disturbance:.2f}$"
        )

    def style_hu_lei_local(disturbance: float) -> LineStyle:
        return LineStyle(
            style=line_style["hu-lei"],
            color=colors["local", disturbance],
            width=1.5,
            marker=markers["hu-lei"],
            label=fr"$\theta={disturbance:.2f}$"
        )

    figure.add_rates(
        i=0,
        run_batches=h0_runs,
        metric=lambda rejection_rates: rejection_rates,
        line_fn=style_h0
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_global_ours,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_ours_global
    )

    figure.add_rates(
        i=1,
        run_batches=h1_runs_global_hu_lei,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_hu_lei_global,
        initial_marker_offset=1
    )

    figure.add_rates(
        i=2,
        run_batches=h1_runs_local_ours,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_ours_local,
    )

    figure.add_rates(
        i=2,
        run_batches=h1_runs_hu_lei_local,
        metric=lambda rejection_rates: 1 - rejection_rates,
        line_fn=style_hu_lei_local,
    )

    figure.add_identity(i=0, sign=+1)
    figure.add_identity(i=1, sign=-1)

    figure.add_legends([
        [
            LineStyle(style="-", color="tab:gray", label="Ours", width=1.5, marker=markers["ours"]),
            None
        ] + [
            LineStyle(
                style="-.",
                color=colors["global", norm],
                label=fr"$\xi={norm:.2f}$",
                width=1.5,
                marker=None
            )
            for norm in sorted(norms)
        ],
        [
            LineStyle(style=":", color="tab:gray", label="Baseline", width=1.5, marker=markers["hu-lei"]),
            None
        ] + [
            LineStyle(
                style="-.",
                color=colors["local", weight],
                label=fr"$\theta={100 * weight:.1f}\%$",
                width=1.5,
                marker=None
            )
            for weight in sorted(weights)
        ]
    ])

    figure.tighten()

    DIR_FIGURES.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURES / "our_vs_baseline.pdf", dpi=500)


def main(figure: int | None = None):
    if figure is None or figure == 2:
        our_vs_baseline()
    if figure is None or figure == 3:
        gaussian_vs_linear()
    if figure is None or figure == 4:
        gaussian_vs_polynomial()
    if figure is None or figure == 6:
        dataset_size()


if __name__ == "__main__":
    set_plot_style("white")
    tyro.cli(main)
