from typing import Self, Any

import jax.numpy as jnp
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing_extensions import NamedTuple

import src.experiments.error_rates.__main__ as error_rates
import src.experiments.error_rates.spec as spec
from src.expyro.experiment import Run
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_RESULTS, DIR_FIGURES_APX


def _plot_line(
        confidence_levels: jnp.ndarray,
        mean: jnp.ndarray,
        bootstrap: jnp.ndarray,
        color: Any,
        line_style: str,
        axis: plt.Axes
):
    bootstrap_mean = bootstrap.mean(axis=-1)
    delta_lo = bootstrap_mean - jnp.quantile(bootstrap, 0.025, axis=-1)
    delta_hi = jnp.quantile(bootstrap, 0.975, axis=-1) - bootstrap_mean

    sns.lineplot(
        x=confidence_levels, y=mean,
        ax=axis,
        color=color,
        linestyle=line_style,
        linewidth=2
    )

    axis.fill_between(confidence_levels, mean - delta_lo, mean + delta_hi, alpha=0.2, color=color)


class PerformancePlot(NamedTuple):
    fig: plt.Figure
    ax_type_1_uniform: plt.Axes
    ax_type_1_local: plt.Axes
    ax_type_2_uniform: plt.Axes
    ax_type_2_local: plt.Axes

    @classmethod
    def default(cls) -> Self:
        fig = plt.figure(figsize=(6.75, 2.25))
        grid = GridSpec(2, 4, figure=fig, height_ratios=[0.15, 0.85])

        ax_fp_uniform = fig.add_subplot(grid[1, 0])
        ax_fp_local = fig.add_subplot(grid[1, 1], sharey=ax_fp_uniform)
        ax_tp_uniform = fig.add_subplot(grid[1, 2], sharey=ax_fp_uniform)
        ax_tp_local = fig.add_subplot(grid[1, 3], sharey=ax_fp_uniform)

        return cls(fig, ax_fp_uniform, ax_fp_local, ax_tp_uniform, ax_tp_local).tidy()

    @classmethod
    def large_legend(cls) -> Self:
        fig = plt.figure(figsize=(6.75, 2.5))
        grid = GridSpec(2, 4, figure=fig, height_ratios=[0.25, 0.75])

        ax_fp_uniform = fig.add_subplot(grid[1, 0])
        ax_fp_local = fig.add_subplot(grid[1, 1], sharey=ax_fp_uniform)
        ax_tp_uniform = fig.add_subplot(grid[1, 2], sharey=ax_fp_uniform)
        ax_tp_local = fig.add_subplot(grid[1, 3], sharey=ax_fp_uniform)

        return cls(fig, ax_fp_uniform, ax_fp_local, ax_tp_uniform, ax_tp_local).tidy()

    def tidy(self) -> Self:
        plt.setp(self.ax_type_1_local.get_yticklabels(), visible=False)
        plt.setp(self.ax_type_2_uniform.get_yticklabels(), visible=False)
        plt.setp(self.ax_type_2_local.get_yticklabels(), visible=False)

        for ax in [self.ax_type_1_uniform, self.ax_type_1_local, self.ax_type_2_uniform, self.ax_type_2_local]:
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylim(-0.05, 1.05)

        self.ax_type_1_uniform.set_title("Uniform type I error", fontdict={"fontsize": TEXT_FONT_SIZE})
        self.ax_type_1_local.set_title("Local type I error", fontdict={"fontsize": TEXT_FONT_SIZE})
        self.ax_type_2_uniform.set_title("Uniform type II error", fontdict={"fontsize": TEXT_FONT_SIZE})
        self.ax_type_2_local.set_title("Local type II error", fontdict={"fontsize": TEXT_FONT_SIZE})

        return self

    def plot_type_1_error[T](
            self,
            false_positive_runs: dict[T, Run[spec.BaseConfig, spec.PositiveRate]],
            colors: dict[T, Any],
            line_style: dict[T, Any],
    ):
        for key, run in false_positive_runs.items():
            _plot_line(
                run.config.confidence_levels(),
                run.result.uniform,
                run.result.bootstrap_uniform_distribution,
                colors[key],
                line_style[key],
                self.ax_type_1_uniform
            )

            _plot_line(
                run.config.confidence_levels(),
                run.result.local,
                run.result.bootstrap_local_distribution,
                colors[key],
                line_style[key],
                self.ax_type_1_local
            )

    def plot_type_2_error[T](
            self,
            true_positive_runs: dict[T, Run[spec.BaseConfig, spec.PositiveRate]],
            colors: dict[T, Any],
            line_style: dict[T, Any],
    ):
        for key, run in true_positive_runs.items():
            _plot_line(
                run.config.confidence_levels(),
                1 - run.result.uniform,
                1 - run.result.bootstrap_uniform_distribution,
                colors[key],
                line_style[key],
                self.ax_type_2_uniform
            )

            _plot_line(
                run.config.confidence_levels(),
                1 - run.result.local,
                1 - run.result.bootstrap_local_distribution,
                colors[key],
                line_style[key],
                self.ax_type_2_local
            )

    def adjust_default(self):
        self.fig.subplots_adjust(top=0.95, left=0.055, right=0.995, bottom=0.18, wspace=0.075)

    def legend_default[T](
            self,
            colors: dict[T, Any],
            line_styles: dict[T, Any],
            labels: dict[T, str],
            order: list[T],
            n_cols: int | None = None
    ):
        assert colors.keys() == line_styles.keys() == labels.keys() == set(order)
        assert len(set(order)) == len(order)

        if n_cols is None:
            n_cols = len(order)

        self.fig.legend(
            handles=[
                mlines.Line2D([], [], color=colors[key], linestyle=line_styles[key], linewidth=2)
                for key in order
            ],
            labels=[labels[key] for key in order],
            loc="upper left",
            ncol=len(order),
            facecolor="white",
            mode="expand",
            ncols=n_cols,
            frameon=False,
            bbox_to_anchor=(0, 1.025, 1, 0),
            fontsize=TEXT_FONT_SIZE,
            handlelength=2,
            handletextpad=0.5,
        )


def make_bright_color_palette[T](palette: str, values: list[T]) -> dict[T, tuple[float, float, float]]:
    palette = sns.color_palette(palette, len(values))

    return {
        value: palette[i]
        for i, value in enumerate(values)
    }


def plot_bootstrap_vs_analytical():
    directory = DIR_RESULTS / "error_rates" / "bootstrap-vs-analytical" / "IIDDataConfig"

    plot = PerformancePlot.default()

    stds = [0, 0.05, 0.1, 0.2]

    colors = make_bright_color_palette("bright", stds)

    line_styles_analytical = {std: ":" for std in stds}
    line_styles_bootstrap = {std: "-" for std in stds}

    plot.plot_type_1_error(
        false_positive_runs={
            std: error_rates.experiment[directory / f"same-fn__bootstrap__std-{std}"]
            for std in stds
        },
        colors=colors,
        line_style=line_styles_bootstrap
    )

    plot.plot_type_1_error(
        false_positive_runs={
            std: error_rates.experiment[directory / f"same-fn__analytical__std-{std}"]
            for std in stds
        },
        colors=colors,
        line_style=line_styles_analytical
    )

    plot.plot_type_2_error(
        true_positive_runs={
            std: error_rates.experiment[directory / f"different-fn__bootstrap__std-{std}"]
            for std in stds
        },
        colors=colors,
        line_style=line_styles_bootstrap
    )

    plot.plot_type_2_error(
        true_positive_runs={
            std: error_rates.experiment[directory / f"different-fn__analytical__std-{std}"]
            for std in stds
        },
        colors=colors,
        line_style=line_styles_analytical
    )

    plot.legend_default(
        colors=colors | {"bootstrap": "black", "analytical": "black"},
        line_styles={std: "-." for std in stds} | {"bootstrap": "-", "analytical": ":"},
        labels={
                   std: rf"$\sigma^2={std}$"
                   for std in stds
               } | {"bootstrap": "Bootstrap", "analytical": "Analytical"},
        order=["analytical", "bootstrap"] + stds
    )

    plot.adjust_default()

    plt.savefig(DIR_FIGURES_APX / "bootstrap-vs-analytical.pdf", format="pdf")


def plot_mixture_noise_intensity():
    directory = DIR_RESULTS / "error_rates" / "mixture-noise-intensity" / "IIDDataConfig"

    plot = PerformancePlot.default()

    intensities = [0.025, 0.05, 0.075, 0.1]

    colors = make_bright_color_palette("bright", intensities)

    line_styles_linear = {intensity: ":" for intensity in intensities}
    line_styles_gaussian = {intensity: "-" for intensity in intensities}

    plot.plot_type_1_error(
        false_positive_runs={
            mean: error_rates.experiment[directory / f"same-noise__mean-{mean}__kernel-y-GaussianKernelConfig"]
            for mean in intensities
        },
        colors=colors,
        line_style=line_styles_gaussian
    )

    plot.plot_type_1_error(
        false_positive_runs={
            mean: error_rates.experiment[directory / f"same-noise__mean-{mean}__kernel-y-LinearKernelConfig"]
            for mean in intensities
        },
        colors=colors,
        line_style=line_styles_linear
    )

    for mean in intensities:
        gaussian = error_rates.experiment[directory / f"different-noise__mean-{mean}__kernel-y-GaussianKernelConfig"]
        linear = error_rates.experiment[directory / f"different-noise__mean-{mean}__kernel-y-LinearKernelConfig"]

        _plot_line(
            gaussian.config.confidence_levels(),
            gaussian.result.uniform,
            gaussian.result.bootstrap_uniform_distribution,
            colors[mean],
            line_styles_gaussian[mean],
            plot.ax_type_2_uniform
        )

        _plot_line(
            gaussian.config.confidence_levels(),
            gaussian.result.local,
            gaussian.result.bootstrap_local_distribution,
            colors[mean],
            line_styles_gaussian[mean],
            plot.ax_type_2_local
        )

        _plot_line(
            linear.config.confidence_levels(),
            linear.result.uniform,
            linear.result.bootstrap_uniform_distribution,
            colors[mean],
            line_styles_linear[mean],
            plot.ax_type_2_uniform
        )

        _plot_line(
            linear.config.confidence_levels(),
            linear.result.local,
            linear.result.bootstrap_local_distribution,
            colors[mean],
            line_styles_linear[mean],
            plot.ax_type_2_local
        )

    plot.ax_type_1_uniform.set_title(
        r"Uniform pos. rate ($\mathbb{P} = \mathbb{Q}$)",
        fontdict={"size": TEXT_FONT_SIZE}
    )

    plot.ax_type_1_local.set_title(
        r"Local pos. rate ($\mathbb{P} = \mathbb{Q}$)",
        fontdict={"size": TEXT_FONT_SIZE}
    )

    plot.ax_type_2_uniform.set_title(
        r"Uniform pos. rate ($\mathbb{P} \neq \mathbb{Q}$)",
        fontdict={"size": TEXT_FONT_SIZE}
    )

    plot.ax_type_2_local.set_title(
        r"Local pos. rate ($\mathbb{P} \neq \mathbb{Q}$)",
        fontdict={"size": TEXT_FONT_SIZE}
    )

    plot.legend_default(
        colors=colors | {"linear": "black", "gaussian": "black"},
        line_styles={intensity: "-." for intensity in intensities} | {"linear": ":", "gaussian": "-"},
        labels={
                   intensity: rf"$\mu={intensity}$"
                   for intensity in intensities
               } | {"linear": "Linear", "gaussian": "Gaussian"},
        order=["linear", "gaussian"] + intensities
    )

    plot.adjust_default()

    plt.savefig(DIR_FIGURES_APX / "mixture-noise-intensity.pdf", format="pdf")


def plot_dataset_size():
    directory = DIR_RESULTS / "error_rates" / "dataset-size"

    plot = PerformancePlot.default()

    sizes = [10, 20, 50, 100]
    colors = make_bright_color_palette("bright", sizes)

    line_styles = {
        10: "-",
        20: "--",
        50: "-.",
        100: "-",
    }

    plot.plot_type_1_error(
        false_positive_runs={
            10: error_rates.experiment[directory / f"same-fn__size-10"],
            20: error_rates.experiment[directory / f"same-fn__size-20"],
            50: error_rates.experiment[directory / f"same-fn__size-50"],
            100: error_rates.experiment[directory / f"same-fn__size-100"],
        },
        colors=colors,
        line_style=line_styles,
    )

    plot.plot_type_2_error(
        true_positive_runs={
            10: error_rates.experiment[directory / f"different-fn__size-10"],
            20: error_rates.experiment[directory / f"different-fn__size-20"],
            50: error_rates.experiment[directory / f"different-fn__size-50"],
            100: error_rates.experiment[directory / f"different-fn__size-100"],
        },
        colors=colors,
        line_style=line_styles,
    )

    plot.legend_default(
        colors=colors,
        line_styles=line_styles,
        labels={
            size: rf"$N={size}$"
            for size in sizes
        },
        order=sizes
    )

    plot.adjust_default()

    plt.savefig(DIR_FIGURES_APX / "dataset-size.pdf", format="pdf")


def plot_disturbance():
    directory = DIR_RESULTS / "error_rates" / "disturbance"

    plot = PerformancePlot.default()

    magnitudes = [0.1, 0.25, 0.5, 0.75, 1]
    colors = make_bright_color_palette("bright", magnitudes)

    line_styles = {
        0.1: "-",
        0.25: "--",
        0.5: "-.",
        0.75: "-",
        1: "--"
    }

    plot.plot_type_1_error(
        false_positive_runs={
            0: error_rates.experiment[directory / f"single-fn"],
        },
        colors={0: "black"},
        line_style={0: "-"},
    )

    plot.plot_type_2_error(
        true_positive_runs={
            magnitude: error_rates.experiment[directory / f"disturbed-fn__magnitude-{magnitude}"]
            for magnitude in magnitudes
        },
        colors=colors,
        line_style=line_styles,
    )

    plot.legend_default(
        colors=colors,
        line_styles=line_styles,
        labels={
            magnitude: rf"$\xi={magnitude}$"
            for magnitude in magnitudes
        },
        order=magnitudes
    )

    plot.adjust_default()

    plt.savefig(DIR_FIGURES_APX / "disturbance.pdf", format="pdf")


def plot_moment_richness():
    directory = DIR_RESULTS / "error_rates" / "moment-richness" / "IIDDataConfig"

    plot = PerformancePlot.large_legend()

    degrees = [1, 2, 3]
    bandwidths = [0.05, 0.1, 0.15]

    line_styles_polynomial = {degree: ":" for degree in degrees}
    line_styles_gaussian = {bandwidth: "-" for bandwidth in bandwidths}

    colors = make_bright_color_palette("bright", degrees + bandwidths)

    plot.plot_type_1_error(
        false_positive_runs={
            bandwidth: error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-{bandwidth}"]
            for bandwidth in bandwidths
        },
        colors=colors,
        line_style=line_styles_gaussian
    )

    plot.plot_type_1_error(
        false_positive_runs={
            degree: error_rates.experiment[directory / f"same-fn__polynomial__degree-{degree}"]
            for degree in degrees
        },
        colors=colors,
        line_style=line_styles_polynomial
    )

    plot.plot_type_2_error(
        true_positive_runs={
            bandwidth: error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-{bandwidth}"]
            for bandwidth in bandwidths
        },
        colors=colors,
        line_style=line_styles_gaussian
    )

    plot.plot_type_2_error(
        true_positive_runs={
            degree: error_rates.experiment[directory / f"different-fn__polynomial__degree-{degree}"]
            for degree in degrees
        },
        colors=colors,
        line_style=line_styles_polynomial
    )

    line_styles = line_styles_polynomial | line_styles_gaussian | {"polynomial": ":", "gaussian": "-"}
    labels = {
        degree: rf"$d={degree}$"
        for degree in degrees
    } | {
        bandwidth: rf"$\gamma^2={bandwidth}$"
        for bandwidth in bandwidths
    } | {
        "polynomial": "Polynomial",
        "gaussian": "Gaussian"
    }

    plot.legend_default(
        colors=colors | {"polynomial": "black", "gaussian": "black"},
        line_styles=line_styles,
        labels=labels,
        order=["polynomial", "gaussian", 1, 0.05, 2, 0.1, 3, 0.15],
        n_cols=4
    )

    plot.adjust_default()

    plt.savefig(DIR_FIGURES_APX / "moment-richness.pdf", format="pdf")


if __name__ == "__main__":
    set_plot_style()

    plot_bootstrap_vs_analytical()
    plot_mixture_noise_intensity()
    plot_dataset_size()
    plot_disturbance()
    plot_moment_richness()
