import jax.numpy as jnp
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib import pyplot as plt

import src.experiments.error_rates.__main__ as error_rates
import src.experiments.error_rates.spec as spec
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_RESULTS, DIR_FIGURES


def main():
    dir_runs = DIR_RESULTS / "error_rates" / "mixture-noise-intensity" / "IIDDataConfig"
    run_gaussian = error_rates.experiment[dir_runs / "different-noise__mean-0.075__kernel-y-GaussianKernelConfig"]
    run_linear = error_rates.experiment[dir_runs / "different-noise__mean-0.075__kernel-y-LinearKernelConfig"]

    config_gaussian: spec.BaseConfig = run_gaussian.config
    config_linear: spec.BaseConfig = run_linear.config
    result_gaussian: spec.PositiveRate = run_gaussian.result
    result_linear: spec.PositiveRate = run_linear.result

    confidence_levels = config_gaussian.confidence_levels()

    assert jnp.all(config_linear.confidence_levels() == confidence_levels)

    fig = plt.figure(figsize=(3.25, 2))
    ax = fig.add_subplot(111)

    ax.legend(
        handles=[
            mlines.Line2D([], [], color="tab:orange", linestyle="-", linewidth=2),
            mlines.Line2D([], [], color="tab:blue", linestyle="-", linewidth=2),
        ],
        labels=[
            rf"Gaussian with $\gamma^2 = {config_gaussian.kernel.x_config.bandwidth}$",
            "Linear",
        ],
        loc="upper left",
        ncol=5,
        facecolor="white",
        mode="expand",
        frameon=False,
        bbox_to_anchor=(0, 1.25, 1, 0),
        fontsize=TEXT_FONT_SIZE,
        handlelength=1,
        handletextpad=0.5,
    )

    def plot_with_ci(result: spec.PositiveRate, color):
        bootstrap_mean = jnp.mean(result.bootstrap_uniform_distribution, axis=-1)
        lo = bootstrap_mean - jnp.quantile(result.bootstrap_uniform_distribution, 0.025, axis=-1)
        hi = jnp.quantile(result.bootstrap_uniform_distribution, 0.975, axis=-1) - bootstrap_mean

        sns.lineplot(
            x=confidence_levels, y=result.uniform,
            ax=ax,
            color=color
        )

        ax.fill_between(
            x=confidence_levels,
            y1=result.uniform - lo,
            y2=result.uniform + hi,
            alpha=0.2,
            color=color
        )

    plot_with_ci(result_gaussian, color="tab:orange")
    plot_with_ci(result_linear, color="tab:blue")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Uniform positive rate")

    plt.tight_layout(pad=0.25, rect=(0, -0.015, 1, 1.045))

    plt.savefig(DIR_FIGURES / "gaussian-vs-linear.pdf", format="pdf")


if __name__ == "__main__":
    set_plot_style()
    main()
