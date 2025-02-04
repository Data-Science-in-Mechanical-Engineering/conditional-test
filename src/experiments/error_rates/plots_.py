from collections.abc import Callable

import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import src.experiments.error_rates.__main__ as error_rates
import src.experiments.error_rates.spec as error_rates_spec
from src.config import KernelConfig, DataConfig
from src.expyro.experiment import Run
from src.figures.util import set_plot_style
from src.util import DIR_RESULTS


def plot_rates[T](
        false_positive_runs: dict[T, Run[error_rates_spec.BaseConfig, error_rates_spec.PositiveRate]],
        true_positive_runs: dict[T, Run[error_rates_spec.BaseConfig, error_rates_spec.PositiveRate]],
        legend_order: list[T],
        legend_symbol: Callable[[T], str],
        legend_cols: int
):
    some_run = next(iter(false_positive_runs.values()))
    confidence_levels = some_run.config.confidence_levels()

    false_positives = {key: run.result for key, run in false_positive_runs.items()}
    true_positives = {key: run.result for key, run in true_positive_runs.items()}

    assert all(jnp.all(run.config.confidence_levels() == confidence_levels) for run in false_positive_runs.values())
    assert false_positives.keys() == true_positives.keys()
    parameters = list(sorted(false_positives.keys()))

    fig = plt.figure(figsize=(10, 8))

    grid = GridSpec(
        figure=fig,
        nrows=3, ncols=2,
        height_ratios=[0.05, 1, 1],
    )

    ax_uniform_fp = fig.add_subplot(grid[1, 0])
    ax_local_fp = fig.add_subplot(grid[2, 0], sharex=ax_uniform_fp)
    ax_uniform_tp = fig.add_subplot(grid[1, 1], sharey=ax_uniform_fp)
    ax_local_tp = fig.add_subplot(grid[2, 1], sharex=ax_uniform_tp, sharey=ax_local_fp)

    ax_uniform_fp.set_ylabel("Uniform type I error")
    ax_local_fp.set_ylabel("Local type I error")
    ax_uniform_tp.set_ylabel("Uniform type II error")
    ax_local_tp.set_ylabel("Local type II error")

    plt.setp(ax_uniform_fp.get_xticklabels(), visible=False)
    plt.setp(ax_uniform_tp.get_yticklabels(), visible=False)
    plt.setp(ax_uniform_tp.get_xticklabels(), visible=False)
    plt.setp(ax_local_tp.get_yticklabels(), visible=False)

    ax_local_fp.set_xlabel(r"$\alpha$")
    ax_local_tp.set_xlabel(r"$\alpha$")

    palette = sns.color_palette("bright", len(parameters))
    line_styles = ["-", "--", "-.", ":"]

    for parameter, color, i in zip(legend_order, palette, range(len(parameters))):
        fp = false_positives[parameter]
        tp = true_positives[parameter]

        def plot_line(mean: jnp.ndarray, bootstrap: jnp.ndarray, axis: plt.Axes):
            bootstrap_mean = bootstrap.mean(axis=-1)
            delta_lo = bootstrap_mean - jnp.quantile(bootstrap, 0.025, axis=-1)
            delta_hi = jnp.quantile(bootstrap, 0.975, axis=-1) - bootstrap_mean

            sns.lineplot(
                x=confidence_levels, y=mean,
                ax=axis,
                label=legend_symbol(parameter),
                color=color,
                linestyle=line_styles[i % len(line_styles)],
                linewidth=3
            )

            axis.fill_between(confidence_levels, mean - delta_lo, mean + delta_hi, alpha=0.2, color=color)

        plot_line(
            mean=fp.uniform,
            bootstrap=fp.bootstrap_uniform_distribution,
            axis=ax_uniform_fp,
        )

        plot_line(
            mean=fp.local,
            bootstrap=fp.bootstrap_local_distribution,
            axis=ax_local_fp,
        )

        plot_line(
            mean=1 - tp.uniform,
            bootstrap=1 - tp.bootstrap_uniform_distribution,
            axis=ax_uniform_tp,
        )

        plot_line(
            mean=1 - tp.local,
            bootstrap=1 - tp.bootstrap_local_distribution,
            axis=ax_local_tp,
        )

    for ax in [ax_uniform_fp, ax_local_fp, ax_uniform_tp, ax_local_tp]:
        ax.set_ylim(-0.025, 1.025)
        ax.get_legend().remove()

    handles, labels = ax_uniform_fp.get_legend_handles_labels()
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")

    legend_axis.legend(
        handles, labels,
        loc="upper center",
        ncol=len(parameters),
        facecolor="white",
        mode="expand",
        ncols=legend_cols,
        frameon=False,
        bbox_to_anchor=(-0.01, 1, 1.02, 0)
    )

    plt.tight_layout(pad=1)
    plt.show()


def plot_rates_mixture_noise_intensity(data_config: type[DataConfig], kernel_config: type[KernelConfig]):
    directory = DIR_RESULTS / "error_rates" / "mixture-noise-intensity" / f"{data_config.__name__}"

    false_positive_runs = {
        0.1: error_rates.experiment[directory / f"same-noise__mean-0.1__kernel-y-{kernel_config.__name__}"],
        0.075: error_rates.experiment[directory / f"same-noise__mean-0.075__kernel-y-{kernel_config.__name__}"],
        0.05: error_rates.experiment[directory / f"same-noise__mean-0.05__kernel-y-{kernel_config.__name__}"],
        0.025: error_rates.experiment[directory / f"same-noise__mean-0.025__kernel-y-{kernel_config.__name__}"],
    }

    true_positive_runs = {
        0.1: error_rates.experiment[directory / f"different-noise__mean-0.1__kernel-y-{kernel_config.__name__}"],
        0.075: error_rates.experiment[directory / f"different-noise__mean-0.075__kernel-y-{kernel_config.__name__}"],
        0.05: error_rates.experiment[directory / f"different-noise__mean-0.05__kernel-y-{kernel_config.__name__}"],
        0.025: error_rates.experiment[directory / f"different-noise__mean-0.025__kernel-y-{kernel_config.__name__}"],
    }

    plot_rates(
        false_positive_runs=false_positive_runs,
        true_positive_runs=true_positive_runs,
        legend_order=[0.025, 0.05, 0.075, 0.1],
        legend_symbol=lambda x: fr"$\mu={x}$",
        legend_cols=4
    )


def plot_moment_richness(data_config: type[DataConfig]):
    directory = DIR_RESULTS / "error_rates" / "moment-richness" / f"{data_config.__name__}"

    false_positive_runs = {
        ("polynomial", 1): error_rates.experiment[directory / f"same-fn__polynomial__degree-1"],
        ("polynomial", 2): error_rates.experiment[directory / f"same-fn__polynomial__degree-2"],
        ("polynomial", 3): error_rates.experiment[directory / f"same-fn__polynomial__degree-3"],
        ("gaussian", 0.05): error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.05"],
        ("gaussian", 0.1): error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.1"],
        ("gaussian", 0.15): error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.15"],
    }

    true_positive_runs = {
        ("polynomial", 1): error_rates.experiment[directory / f"different-fn__polynomial__degree-1"],
        ("polynomial", 2): error_rates.experiment[directory / f"different-fn__polynomial__degree-2"],
        ("polynomial", 3): error_rates.experiment[directory / f"different-fn__polynomial__degree-3"],
        ("gaussian", 0.05): error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.05"],
        ("gaussian", 0.1): error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.1"],
        ("gaussian", 0.15): error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.15"],
    }

    def legend_symbol(identifier: tuple[str, float]) -> str:
        kernel_type, parameter = identifier

        if kernel_type == "polynomial":
            return fr"Polynomial with $d={parameter}$"
        elif kernel_type == "gaussian":
            return fr"Gaussian with $\gamma^2={parameter}$"
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    plot_rates(
        false_positive_runs=false_positive_runs,
        true_positive_runs=true_positive_runs,
        legend_order=[
            ("polynomial", 1), ("gaussian", 0.05),
            ("polynomial", 2), ("gaussian", 0.1),
            ("polynomial", 3), ("gaussian", 0.15),
        ],
        legend_symbol=legend_symbol,
        legend_cols=3
    )


def plot_bootstrap_performance(data_config: type[DataConfig]):
    directory = DIR_RESULTS / "error_rates" / "bootstrap-vs-analytical" / f"{data_config.__name__}"

    false_positive_runs = {
        0: error_rates.experiment[directory / f"same-fn__bootstrap__std-0"],
        0.01: error_rates.experiment[directory / f"same-fn__bootstrap__std-0.01"],
        0.02: error_rates.experiment[directory / f"same-fn__bootstrap__std-0.02"],
        0.05: error_rates.experiment[directory / f"same-fn__bootstrap__std-0.05"],
        0.1: error_rates.experiment[directory / f"same-fn__bootstrap__std-0.1"],
        0.2: error_rates.experiment[directory / f"same-fn__bootstrap__std-0.2"],
    }

    true_positive_runs = {
        0: error_rates.experiment[directory / f"different-fn__bootstrap__std-0"],
        0.01: error_rates.experiment[directory / f"different-fn__bootstrap__std-0.01"],
        0.02: error_rates.experiment[directory / f"different-fn__bootstrap__std-0.02"],
        0.05: error_rates.experiment[directory / f"different-fn__bootstrap__std-0.05"],
        0.1: error_rates.experiment[directory / f"different-fn__bootstrap__std-0.1"],
        0.2: error_rates.experiment[directory / f"different-fn__bootstrap__std-0.2"],
    }

    plot_rates(
        false_positive_runs=false_positive_runs,
        true_positive_runs=true_positive_runs,
        legend_order=[0, 0.01, 0.02, 0.05, 0.1, 0.2],
        legend_symbol=lambda x: fr"$\sigma={x}$",
        legend_cols=6
    )


def plot_analytical_performance(data_config: type[DataConfig]):
    directory = DIR_RESULTS / "error_rates" / "bootstrap-vs-analytical" / f"{data_config.__name__}"

    false_positive_runs = {
        0: error_rates.experiment[directory / f"same-fn__analytical__std-0"],
        0.01: error_rates.experiment[directory / f"same-fn__analytical__std-0.01"],
        0.02: error_rates.experiment[directory / f"same-fn__analytical__std-0.02"],
        0.05: error_rates.experiment[directory / f"same-fn__analytical__std-0.05"],
        0.1: error_rates.experiment[directory / f"same-fn__analytical__std-0.1"],
        0.2: error_rates.experiment[directory / f"same-fn__analytical__std-0.2"],
    }

    true_positive_runs = {
        0: error_rates.experiment[directory / f"different-fn__analytical__std-0"],
        0.01: error_rates.experiment[directory / f"different-fn__analytical__std-0.01"],
        0.02: error_rates.experiment[directory / f"different-fn__analytical__std-0.02"],
        0.05: error_rates.experiment[directory / f"different-fn__analytical__std-0.05"],
        0.1: error_rates.experiment[directory / f"different-fn__analytical__std-0.1"],
        0.2: error_rates.experiment[directory / f"different-fn__analytical__std-0.2"],
    }

    plot_rates(
        false_positive_runs=false_positive_runs,
        true_positive_runs=true_positive_runs,
        legend_order=[0, 0.01, 0.02, 0.05, 0.1, 0.2],
        legend_symbol=lambda x: fr"$\sigma={x}$",
        legend_cols=6
    )


def plot_dataset_size():
    directory = DIR_RESULTS / "error_rates" / "dataset-size"

    false_positive_runs = {
        10: error_rates.experiment[directory / f"same-fn__size-10"],
        20: error_rates.experiment[directory / f"same-fn__size-20"],
        50: error_rates.experiment[directory / f"same-fn__size-50"],
        100: error_rates.experiment[directory / f"same-fn__size-100"],
    }

    true_positive_runs = {
        10: error_rates.experiment[directory / f"different-fn__size-10"],
        20: error_rates.experiment[directory / f"different-fn__size-20"],
        50: error_rates.experiment[directory / f"different-fn__size-50"],
        100: error_rates.experiment[directory / f"different-fn__size-100"],
    }

    plot_rates(
        false_positive_runs=false_positive_runs,
        true_positive_runs=true_positive_runs,
        legend_order=[10, 20, 50, 100],
        legend_symbol=lambda x: fr"$N={x}$",
        legend_cols=4
    )


if __name__ == "__main__":
    set_plot_style()

    plot_rates_mixture_noise_intensity(error_rates.IIDDataConfig, error_rates.GaussianKernelConfig)
    plot_rates_mixture_noise_intensity(error_rates.IIDDataConfig, error_rates.LinearKernelConfig)

    plot_moment_richness(error_rates.IIDDataConfig)

    plot_bootstrap_performance(error_rates.IIDDataConfig)
    plot_analytical_performance(error_rates.IIDDataConfig)

    plot_dataset_size()
