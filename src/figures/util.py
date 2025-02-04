import matplotlib
import seaborn as sns

TEXT_FONT_SIZE = 10  # pt


def set_plot_style(grid: str = "whitegrid"):
    sns.set_style(grid)
    matplotlib.rcParams.update({"font.size": TEXT_FONT_SIZE})
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
