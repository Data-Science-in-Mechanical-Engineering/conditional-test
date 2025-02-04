import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import src.experiments.error_rates.__main__ as error_rates
import src.experiments.error_rates.spec as spec
from src.expyro.experiment import Run
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_RESULTS, DIR_FIGURES

import matplotlib.lines as mlines


def main():
    directory = DIR_RESULTS / "error_rates" / "moment-richness" / "IIDDataConfig"

    false_positives_polynomial = {
        1: error_rates.experiment[directory / f"same-fn__polynomial__degree-1"],
        2: error_rates.experiment[directory / f"same-fn__polynomial__degree-2"],
        3: error_rates.experiment[directory / f"same-fn__polynomial__degree-3"],
    }

    false_positives_gaussian = {
        0.05: error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.05"],
        0.1: error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.1"],
        0.15: error_rates.experiment[directory / f"same-fn__gaussian__bandwidth-0.15"],
    }

    true_positives_polynomial = {
        1: error_rates.experiment[directory / f"different-fn__polynomial__degree-1"],
        2: error_rates.experiment[directory / f"different-fn__polynomial__degree-2"],
        3: error_rates.experiment[directory / f"different-fn__polynomial__degree-3"],
    }

    true_positives_gaussian = {
        0.05: error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.05"],
        0.1: error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.1"],
        0.15: error_rates.experiment[directory / f"different-fn__gaussian__bandwidth-0.15"],
    }

    palette = sns.color_palette("bright", 6)

    polynomial_colors = {
        1: palette[0],
        2: palette[1],
        3: palette[2],
    }

    gaussian_colors = {
        0.05: palette[3],
        0.1: palette[4],
        0.15: palette[5],
    }

    fig = plt.figure(figsize=(3.25, 2.5))
    grid = GridSpec(nrows=2, ncols=1, figure=fig)

    ax_false_positives = fig.add_subplot(grid[0])
    ax_false_negatives = fig.add_subplot(grid[1], sharex=ax_false_positives, sharey=ax_false_positives)

    ax_false_positives.legend(
        handles=[
            mlines.Line2D([], [], color="tab:gray", linestyle="-", linewidth=1),
            mlines.Line2D([], [], color="tab:gray", linestyle="--", linewidth=1),
            mlines.Line2D([], [], color=polynomial_colors[1], linestyle="-", linewidth=1),
            mlines.Line2D([], [], color=gaussian_colors[0.05], linestyle="--", linewidth=1),
            mlines.Line2D([], [], color=polynomial_colors[2], linestyle="-", linewidth=1),
            mlines.Line2D([], [], color=gaussian_colors[0.1], linestyle="--", linewidth=1),
            mlines.Line2D([], [], color=polynomial_colors[3], linestyle="-", linewidth=1),
            mlines.Line2D([], [], color=gaussian_colors[0.15], linestyle="--", linewidth=1),
        ],
        labels=[
            "Polyn.",
            "Gauss.",
            r"$d=1$",
            r"$\gamma^2=0.05$",
            r"$d=2$",
            r"$\gamma^2=0.1$",
            r"$d=3$",
            r"$\gamma^2=0.15$",
        ],
        loc="upper left",
        ncol=4,
        facecolor="white",
        mode="expand",
        frameon=False,
        bbox_to_anchor=(-0.2, 1.8, 1.25, 0),
        fontsize=TEXT_FONT_SIZE,
        handlelength=1,
        handletextpad=0.25,
    )

    def plot(run: Run[spec.BaseConfig, spec.PositiveRate], ax, line_style, color):
        if ax is ax_false_positives:
            rate = run.result.uniform
            bootstrap = run.result.bootstrap_uniform_distribution
        elif ax is ax_false_negatives:
            rate = 1 - run.result.uniform
            bootstrap = 1 - run.result.bootstrap_uniform_distribution
        else:
            raise ValueError("Unknown axis")

        lo = bootstrap.mean(axis=-1) - jnp.quantile(bootstrap, 0.025, axis=-1)
        hi = jnp.quantile(bootstrap, 0.975, axis=-1) - bootstrap.mean(axis=-1)

        sns.lineplot(
            x=run.config.confidence_levels(), y=rate,
            ax=ax,
            linestyle=line_style, color=color
        )

        ax.fill_between(
            run.config.confidence_levels(), rate - lo, rate + hi,
            alpha=0.2
        )

    for degree, run in false_positives_polynomial.items():
        plot(run, ax_false_positives, line_style="-", color=polynomial_colors[degree])

    for bandwidth, run in false_positives_gaussian.items():
        plot(run, ax_false_positives, line_style="--", color=gaussian_colors[bandwidth])

    for degree, run in true_positives_polynomial.items():
        plot(run, ax_false_negatives, line_style="-", color=polynomial_colors[degree])

    for bandwidth, run in true_positives_gaussian.items():
        plot(run, ax_false_negatives, line_style="--", color=gaussian_colors[bandwidth])

    plt.setp(ax_false_positives.get_xticklabels(), visible=False)

    ax_false_positives.set_ylim(-0.025, 1)

    ax_false_positives.set_ylabel("Type I error")

    ax_false_negatives.set_xlabel(r"$\alpha$")
    ax_false_negatives.set_ylabel("Type II error")

    plt.tight_layout(pad=0.25, rect=(0, -0.02, 1, 1.08))

    plt.savefig(DIR_FIGURES / "gaussian-vs-polynomial.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    set_plot_style()
    main()
