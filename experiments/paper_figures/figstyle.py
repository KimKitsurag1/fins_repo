"""Shared style of the paper's figures: palette (validated categorical slots 1-3 on white), ink tokens, rcParams."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

INK, SECOND, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WIDTH = 4.8                                                # inches; placed at \textwidth


def setup():
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
                         "mathtext.sf": "Inter", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "xtick.major.size": 2, "ytick.major.size": 2,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.1,
                         "lines.markersize": 3.2, "legend.frameon": False, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})


def tidy(ax, grid="y"):
    """Hairline grid behind the data, no top or right spine."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
