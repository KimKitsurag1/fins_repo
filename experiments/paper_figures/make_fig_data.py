"""Figure: the first evaluation trajectory of seed 42 for each equation (space-time, scaled by its maximum |u|), read
from the registered runs in FINS_RESULTS_DIR, which store the evaluation data they were scored on.

Run:  python make_fig_data.py [--out fig_data.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import figstyle as fs

R = os.environ.get("FINS_RESULTS_DIR", "results")
PANELS = (("Burgers", "burgers_runs/clean/kreider-sheet_burgers_s42_hard.npz", r"$2\pi$"),
          ("heat", "noise_weak_runs/clean/weak_heat_s42.npz", "$1$"),
          ("KdV", "noise_weak_runs/clean/weak_kdv_s42.npz", r"$2\pi$"),
          ("Benjamin–Ono", "bo_runs/clean/weak_bo_s42.npz", r"$2\pi$"),
          ("Kuramoto–\nSivashinsky", "ks_runs/clean/weak_ks_s42.npz", "$22$"))
CMAP = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#a82a29"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fig_data.pdf")
    args = ap.parse_args()
    fs.setup()
    fig, axes = plt.subplots(1, 5, figsize=(fs.WIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45],
                                                                         "wspace": 0.62})
    for ax, (title, path, xlab) in zip(axes, PANELS):
        f = np.load(os.path.join(R, path), allow_pickle=True)
        u, xs, ts = f["u"][0], f["xs"][0], f["ts"][0]
        dx = xs[1] - xs[0]
        im = ax.imshow(u.T / np.abs(u).max(), origin="lower", aspect="auto", cmap=CMAP, vmin=-1, vmax=1,
                       extent=(xs[0], xs[-1] + dx, ts[0], ts[-1]), interpolation="nearest", rasterized=True)
        ax.set_xticks([xs[0], xs[-1] + dx])
        ax.set_xticklabels(["$0$", xlab])
        ax.set_yticks([ts[0], ts[-1]])
        ax.set_yticklabels(["$0$", f"${ts[-1]:.3g}$"])
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", color=fs.INK, linespacing=1.0)
        ax.set_xlabel("$x$", labelpad=-4)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("$t$", labelpad=-4)
    cb = fig.colorbar(im, ax=list(axes), orientation="vertical", fraction=0.025, pad=0.02, aspect=18)
    cb.set_ticks([-1, 0, 1])
    cb.set_label(r"$u\,/\max|u|$", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5)
    fig.savefig(args.out, dpi=300)
    print("saved", args.out)


if __name__ == "__main__":
    main()
