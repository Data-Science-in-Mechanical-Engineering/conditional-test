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

    palette = sns.color_palette("bright", 4)

    colors = {
        10: palette[0],
        20: palette[1],
        50: palette[2],
        100: palette[3],
    }

    line_styles = {
        10: "-",
        20: "--",
        50: "-.",
        100: ":",
    }

    fig = plt.figure(figsize=(3.25, 2.5))
    grid = GridSpec(nrows=2, ncols=1, figure=fig)

    ax_false_positives = fig.add_subplot(grid[0])
    ax_false_negatives = fig.add_subplot(grid[1], sharex=ax_false_positives, sharey=ax_false_positives)

    ax_false_positives.legend(
        handles=[
            mlines.Line2D([], [], color=colors[10], linestyle=line_styles[10], linewidth=1),
            mlines.Line2D([], [], color=colors[20], linestyle=line_styles[20], linewidth=1),
            mlines.Line2D([], [], color=colors[50], linestyle=line_styles[50], linewidth=1),
            mlines.Line2D([], [], color=colors[100], linestyle=line_styles[100], linewidth=1),
        ],
        labels=[
            r"$N=10$",
            r"$N=20$",
            r"$N=50$",
            r"$N=100$",
        ],
        loc="upper left",
        ncol=4,
        facecolor="white",
        mode="expand",
        frameon=False,
        bbox_to_anchor=(-0.05, 1.4, 1.1, 0.),
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

    for size, run in false_positive_runs.items():
        plot(run, ax_false_positives, line_styles[size], colors[size])

    for size, run in true_positive_runs.items():
        plot(run, ax_false_negatives, line_styles[size], colors[size])

    plt.setp(ax_false_positives.get_xticklabels(), visible=False)

    ax_false_positives.set_ylim(-0.025, 1)

    ax_false_positives.set_ylabel("Type I error")

    ax_false_negatives.set_xlabel(r"$\alpha$")
    ax_false_negatives.set_ylabel("Type II error")

    plt.tight_layout(pad=0.25, rect=(0, -0.02, 1, 1.03))

    plt.savefig(DIR_FIGURES / "dataset_size.pdf", format="pdf")


if __name__ == "__main__":
    set_plot_style()
    main()
