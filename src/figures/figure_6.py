import seaborn as sns
from matplotlib import pyplot as plt

import matplotlib.lines as mlines

import src.experiments.monitoring.__main__ as monitoring
import src.experiments.monitoring.spec as spec
from src.expyro.experiment import Run
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_FIGURES


def main():
    runs = {
        0.25: monitoring.main["disturbance-0.25"],
        1: monitoring.main["disturbance-1"],
        0.5: monitoring.main["disturbance-0.5"],
        0.75: monitoring.main["disturbance-0.75"],
    }

    palette = sns.color_palette("rocket", 4)

    run_colors = {
        0.25: palette[0],
        0.5: palette[1],
        0.75: palette[2],
        1: palette[3],
    }

    run_line_styles = {
        0.25: "-",
        0.5: "--",
        0.75: "-.",
        1: "-",
    }

    fig = plt.figure(figsize=(3.25, 2.5))
    ax = fig.add_subplot(111)

    ax.legend(
        handles=[
            mlines.Line2D([], [], color=run_colors[0.25], linestyle=run_line_styles[0.25], linewidth=1),
            mlines.Line2D([], [], color=run_colors[0.5], linestyle=run_line_styles[0.5], linewidth=1),
            mlines.Line2D([], [], color=run_colors[0.75], linestyle=run_line_styles[0.75], linewidth=1),
            mlines.Line2D([], [], color=run_colors[1], linestyle=run_line_styles[1], linewidth=1),
        ],
        labels=[
            rf"$\xi=0.25$",
            rf"$\xi=0.50$",
            rf"$\xi=0.75$",
            rf"$\xi=1.0$",
        ],
        loc="upper left",
        ncol=4,
        facecolor="white",
        mode="expand",
        frameon=False,
        bbox_to_anchor=(-0.04, 1.2, 1.08, 0),
        fontsize=TEXT_FONT_SIZE,
        handlelength=1,
        handletextpad=0.25,
    )

    def plot(run: Run[spec.Config, spec.Result], color, line_style):
        result = run.result

        sns.lineplot(
            (result.cmmd_windows / result.threshold_windows).max(axis=-1),
            ax=ax,
            color=color,
            linestyle=line_style
        )

    for disturbance, run in runs.items():
        plot(run, run_colors[disturbance], run_line_styles[disturbance])

    ax.axvline(x=5 * 20 - 20, color="tab:gray", linestyle="dotted", linewidth=0.5)
    ax.axvline(x=5 * 20, color="tab:gray", linestyle="dotted", linewidth=0.5)
    ax.axhline(y=1, color="tab:gray", linestyle="dotted", linewidth=0.5)

    ax.annotate(
        "Change occurs",
        xy=(81, 3.9),
        xytext=(31, 3.9),
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="fancy",
            lw=0.1,
            color="black",
        ),
        fontsize=TEXT_FONT_SIZE,
        color="black",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )

    ax.annotate(
        "Window done\nadapting",
        xy=(101, 3.1),
        xytext=(29, 3.1),
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="fancy",
            lw=0.1,
            color="black",
        ),
        fontsize=TEXT_FONT_SIZE,
        color="black",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )

    ax.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
    ax.minorticks_on()

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Max. ratio")

    plt.tight_layout(pad=0.25, rect=(0, -0.02, 1, 1.04))

    plt.savefig(DIR_FIGURES / "monitoring.pdf", format="pdf")
    # plt.show()


if __name__ == "__main__":
    set_plot_style("white")
    main()
