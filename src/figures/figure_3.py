import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import src.experiments.example_2d.__main__ as example_2d
import src.experiments.example_2d.spec as spec
from src.figures.util import set_plot_style, TEXT_FONT_SIZE
from src.util import DIR_FIGURES


def main():
    run = example_2d.main["res-1"]
    config: spec.Config = run.config
    result: spec.Result = run.result
    max_scale = max(float(result.thresholds_iid.max()), float(result.thresholds_rotation.max()))

    fig = plt.figure(figsize=(3.25, 2))
    grid = GridSpec(nrows=2, ncols=4, height_ratios=[0.1, 1])

    ax_legend_rejection = fig.add_subplot(grid[0, 2:])
    ax_legend_threshold = fig.add_subplot(grid[0, :2])
    ax_iid = fig.add_subplot(grid[1, :2])
    ax_rotation = fig.add_subplot(grid[1, 2:])

    ax_legend_rejection.axis("off")
    ax_legend_threshold.axis("off")

    gradient = np.ones((1, 256))

    ax_legend_rejection.imshow(gradient, extent=(0, 0.12, 0.4, 0.6), aspect="auto", cmap="inferno", alpha=0.15)
    ax_legend_rejection.text(0.15, 0.48, "Rejection region", fontsize=TEXT_FONT_SIZE, verticalalignment="center")
    ax_legend_rejection.set_xlim(0, 1)

    gradient = np.linspace(0, max_scale, 256).reshape(1, -1)

    ax_legend_threshold.imshow(gradient, extent=(0, 0.12, 0.4, 0.6), aspect="auto", cmap="inferno")
    ax_legend_threshold.text(0.15, 0.48, "Threshold", fontsize=TEXT_FONT_SIZE, verticalalignment="center")
    ax_legend_threshold.set_xlim(0, 1)

    def plot_thresholds(cmmd: jnp.ndarray, threshold: jnp.ndarray, ax):
        cmmd = cmmd.reshape(config.resolution, config.resolution)
        threshold = threshold.reshape(config.resolution, config.resolution)

        sns.heatmap(
            threshold,
            ax=ax,
            vmin=0, vmax=max_scale,
            xticklabels=False, yticklabels=False,
            cmap="inferno", cbar=False,
            linewidths=0.0, linecolor="none",
            rasterized=True
        )

        mask = cmmd <= threshold
        overlay = np.ones_like(cmmd)
        overlay[mask] = np.nan

        sns.heatmap(
            overlay,
            ax=ax,
            cmap="Greys",
            alpha=0.3,
            xticklabels=False, yticklabels=False,
            cbar=False,
            linewidths=0.0, linecolor="none",
            rasterized=True
        )

        border_mask = np.isfinite(overlay)
        ax.contour(border_mask, colors="white", linewidths=1.5, levels=[0.5])

    plot_thresholds(result.cmmd_iid, result.thresholds_iid, ax_iid)
    plot_thresholds(result.cmmd_rotation, result.thresholds_rotation, ax_rotation)

    plt.tight_layout(pad=0.25)

    plt.savefig(DIR_FIGURES / "iid-vs-rotation.pdf", format="pdf")
    # plt.show()


if __name__ == "__main__":
    set_plot_style()
    main()
