"""Figure: the lowest eigenvalues of the weak-form pencil through the exact operators (heat, KdV, Benjamin-Ono,
Kuramoto-Sivashinsky), median over seeds 42-51 per noise level (in percent of each trajectory's standard deviation).
Reads the registered runs in FINS_RESULTS_DIR.

Run:  python make_fig_spectra.py [--out fig_spectra.pdf]
"""

import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

R = os.environ.get("FINS_RESULTS_DIR", "results")
PANELS = (("heat (class)", "noise_weak_runs", "heat", 10, 14), ("KdV", "noise_weak_runs", "kdv", 4, 10),
          ("Benjamin–Ono", "bo_runs", "bo", 4, 10), ("Kuramoto–Sivashinsky", "ks_runs", "ks", 3, 10))
CONDS = (("clean", "clean"), ("n1e-2", "noise 1%"), ("n3e-2", "noise 3%"), ("n5e-2", "noise 5%"),
         ("n1e-1", "noise 10%"))
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")   # ordinal blue, validated (light surface)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
FLOOR = 1e-16


def spectra(runs, pde, cond, show):
    """Sorted eigenvalues [seeds, show] of every weak run of a condition."""
    files = sorted(glob.glob(os.path.join(R, runs, cond, f"weak_{pde}_s*.npz")))
    return np.array([np.sort(np.load(f)["lam"])[:show] for f in files]), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fig_spectra.pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
                         "mathtext.sf": "Inter", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK})
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.9), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    handles = {}
    for ax, (title, runs, pde, k, show) in zip(axes, PANELS):
        i = np.arange(1, show + 1)
        ax.axvspan(0.5, k + 0.5, color="#f0efec", lw=0, zorder=0)
        ax.axvline(k + 0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(k + 0.7, 1e-13, f"$k={k}$", color=INK, fontsize=6, va="center")
        for (cond, lab), col in zip(CONDS, RAMP):
            lam, n = spectra(runs, pde, cond, show)
            if not n:
                continue
            med = np.maximum(np.median(lam, axis=0), FLOOR)
            h, = ax.plot(i, med, color=col, lw=1.1, marker="o", ms=2.6, mec="white", mew=0.4, zorder=3)
            handles[lab] = h
        ax.set_yscale("log")
        ax.set_ylim(3e-17, 3)
        ax.set_xlim(0.5, show + 0.5)
        ax.set_xticks(list(range(1, show + 1, 4 if show > 10 else 3)))
        ax.set_title(title, fontsize=6.5, color=INK, pad=3)
        ax.set_xlabel("index $i$")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.tick_params(length=2, width=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("$\\lambda_i$ (median over 10 seeds)")
    axes[0].set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    fig.legend([handles[lab] for _, lab in CONDS if lab in handles], [lab for _, lab in CONDS if lab in handles],
               loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
