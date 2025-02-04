from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import src.experiments.example_1d.__main__ as example_1d
import src.experiments.example_1d.spec as spec
from src.experiments.example_1d.plots import plot_estimate, plot_rejection_intervals, plot_legend
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_FIGURES


def main():
    run = example_1d.main["res-1"]
    config: spec.Config = run.config
    result: spec.Result = run.result

    fig = plt.figure(figsize=(3.25, 2.5))
    grid_spec = GridSpec(nrows=2, ncols=1)

    ax_analytical = fig.add_subplot(grid_spec[0])
    ax_bootstrap = fig.add_subplot(grid_spec[1], sharex=ax_analytical, sharey=ax_analytical)

    plot_legend(ax_analytical)

    state_space = config.state_space()
    flat_states = state_space.flatten()

    plot_estimate(
        states=flat_states,
        values=result.values_1, estimated_values=result.estimated_values_1,
        dataset_xs=result.dataset_1.xs, dataset_ys=result.dataset_1.ys,
        beta=float(result.beta_1_analytical), sigmas=result.sigmas_1,
        ax=ax_analytical,
        color="tab:orange"
    )

    plot_estimate(
        states=flat_states,
        values=result.values_2, estimated_values=result.estimated_values_2,
        dataset_xs=result.dataset_2.xs, dataset_ys=result.dataset_2.ys,
        beta=float(result.beta_2_analytical), sigmas=result.sigmas_2,
        ax=ax_analytical,
        color="tab:blue"
    )

    plot_estimate(
        states=flat_states,
        values=result.values_1, estimated_values=result.estimated_values_1,
        dataset_xs=result.dataset_1.xs, dataset_ys=result.dataset_1.ys,
        beta=float(result.beta_1_botstrap), sigmas=result.sigmas_1,
        ax=ax_bootstrap,
        color="tab:orange"
    )

    plot_estimate(
        states=flat_states,
        values=result.values_2, estimated_values=result.estimated_values_2,
        dataset_xs=result.dataset_2.xs, dataset_ys=result.dataset_2.ys,
        beta=float(result.beta_2_bootstrap), sigmas=result.sigmas_2,
        ax=ax_bootstrap,
        color="tab:blue"
    )

    ax_analytical.text(
        x=0.02, y=0.075,
        s="Analytical",
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        transform=ax_analytical.transAxes,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )

    ax_bootstrap.text(
        x=0.02, y=0.075,
        s="Bootstrap",
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        transform=ax_bootstrap.transAxes,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )

    for ax in [ax_analytical, ax_bootstrap]:
        ax.margins(x=0)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    axis_min = min(float(result.values_1.min()), float(result.values_2.min())) - 0.2
    axis_max = max(float(result.values_1.max()), float(result.values_2.max())) + 0.2
    ax_analytical.set_ylim(axis_min, axis_max)

    plot_rejection_intervals(
        rejection_intervals=result.rejection_intervals(state_space, result.thresholds_analytical()),
        axis_min=axis_min,
        fig=fig,
        ax=ax_analytical
    )

    plot_rejection_intervals(
        rejection_intervals=result.rejection_intervals(state_space, result.thresholds_bootstrap()),
        axis_min=axis_min,
        fig=fig,
        ax=ax_bootstrap
    )

    plt.tight_layout(pad=0.25)

    plt.savefig(DIR_FIGURES / "example_1d.pdf", format="pdf")

if __name__ == "__main__":
    set_plot_style()
    main()
