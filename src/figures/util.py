from typing import Hashable, Iterable

import seaborn as sns
from matplotlib.collections import LineCollection

type Color = str | tuple[float, float, float]

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "x", "s", "D", "^"]


def make_color_palette[T: Hashable](palette: str, values: Iterable[T]) -> dict[T, Color]:
    palette = sns.color_palette(palette, len(values))

    return {
        value: palette[i]
        for i, value in enumerate(values)
    }


def make_line_style_palette[T: Hashable](values: Iterable[T], styles: list[str] | None = None) -> dict[T, Color]:
    if styles is None:
        styles = LINE_STYLES

    return {
        value: styles[i % len(styles)]
        for i, value in enumerate(values)
    }


def make_marker_palette[T: Hashable](values: Iterable[T]) -> dict[T, str]:
    return {
        value: MARKERS[i % len(MARKERS)]
        for i, value in enumerate(values)
    }


class MulticolorPatch:
    colors: list[str]
    linewidth: float
    line_style: str
    round: bool

    def __init__(self, colors: list[str], linewidth: float, line_style: str, round: bool):
        self.colors = colors
        self.linewidth = linewidth
        self.line_style = line_style
        self.round = round

    def __len__(self):
        return len(self.colors)


class MulticolorPatchHandler:
    def legend_artist(self, _, orig_handle, __, handlebox):
        width, height = handlebox.width, handlebox.height

        segment_width = width / len(orig_handle)

        patch = LineCollection(
            segments=[
                ((i * segment_width, height / 2), ((i + 1) * segment_width, height / 2))
                for i in range(len(orig_handle))
            ],
            linestyles=orig_handle.line_style,
            colors=orig_handle.colors,
            linewidth=orig_handle.linewidth
        )

        if orig_handle.round:
            patch.set_capstyle("round")

        handlebox.add_artist(patch)
        return patch
