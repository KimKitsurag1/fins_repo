"""Figure: the method in one picture. (a) A symmetry commutes with the solution operator (KdV soliton at t = 0,
Galilean boost x -> x + eps t, u -> u + eps/6, in closed form); its derivative in eps is the transport of the
characteristic.
(b) The lowest eigenvalues of the weak-form eigenvalue problem through a learned KdV stepper (seed 42, clean data), read from the
reported run in FINS_RESULTS_DIR.

Run:  python make_fig_method.py [--out fig_method.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

import figstyle as fs

R = os.environ.get("FINS_RESULTS_DIR", "results")
RUN = os.path.join("noise_weak_runs", "clean", "weak-learned_kdv_s42.npz")
C, TAU, EPS, SHOW, K = 2.0, 2.5, 1.5, 12, 4
X = np.linspace(-5.0, 13.0, 600)


def soliton(x0, b=0.0):
    """KdV soliton of speed C centred at x0 on the background b."""
    return 0.5 * C / np.cosh(0.5 * np.sqrt(C) * (X - x0)) ** 2 + b


def curve(fig, rect, y, ref=None, label=""):
    ax = fig.add_axes(rect)
    if ref is not None:
        ax.plot(X, ref, color=fs.MUTED, lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    ax.plot(X, y, color=fs.BLUE, lw=1.1, zorder=2)
    ax.axhline(0, color=fs.GRID, lw=0.5, zorder=0)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(-0.12, 1.35)
    ax.axis("off")
    ax.text(0.0, 1.0, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=fs.INK)
    return ax


def arrow(fig, a, b, text, offset):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                                       color=fs.SECOND, lw=0.7))
    fig.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, ha="center", va="center",
             fontsize=7, color=fs.INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fig_method.pdf")
    args = ap.parse_args()
    fs.setup()
    fig = plt.figure(figsize=(fs.WIDTH, 2.25))
    lift, shift = EPS / 6.0, EPS * TAU
    w, h = 0.22, 0.22
    curve(fig, (0.03, 0.60, w, h), soliton(0.0), label=r"$u(\cdot,0)$")
    curve(fig, (0.33, 0.60, w, h), soliton(C * TAU), label=r"$u(\cdot,\tau)$")
    curve(fig, (0.03, 0.20, w, h), soliton(0.0, lift), ref=soliton(0.0), label=r"$(g_\varepsilon\cdot u)(\cdot,0)$")
    curve(fig, (0.33, 0.20, w, h), soliton(C * TAU + shift, lift), ref=soliton(C * TAU),
          label=r"$(g_\varepsilon\cdot u)(\cdot,\tau)$")
    arrow(fig, (0.255, 0.71), (0.325, 0.71), r"$\Phi_\tau$", (0, 0.045))
    arrow(fig, (0.255, 0.31), (0.325, 0.31), r"$\Phi_\tau$", (0, 0.045))
    arrow(fig, (0.14, 0.585), (0.14, 0.495), r"$g_\varepsilon$", (0.035, 0))
    arrow(fig, (0.44, 0.585), (0.44, 0.495), r"$g_\varepsilon$", (0.035, 0))
    fig.text(0.29, 0.05, r"derivative at $\varepsilon=0$:  $Q(\cdot,\tau)=D\Phi_\tau[u(\cdot,0)]\,Q(\cdot,0)$",
             ha="center", va="center", fontsize=7, color=fs.INK)
    fig.text(0.005, 0.97, "(a)", fontsize=7, color=fs.INK, va="top", fontweight="bold")
    fig.text(0.30, 0.97, "a symmetry commutes with the solution operator", fontsize=6.5, color=fs.SECOND,
             va="top", ha="center")
    f = np.load(os.path.join(R, RUN), allow_pickle=True)
    lam = np.sort(f["lam"])[:SHOW]
    ax = fig.add_axes((0.70, 0.17, 0.28, 0.66))
    i = np.arange(1, SHOW + 1)
    ax.plot(i[:K], lam[:K], "o", ms=3.6, color=fs.BLUE, mec="white", mew=0.4, zorder=3)
    ax.plot(i[K:], lam[K:], "o", ms=3.6, color=fs.MUTED, mec="white", mew=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0.3, SHOW + 0.7)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.annotate("", xy=(K + 0.5, lam[K]), xytext=(K + 0.5, lam[K - 1]),
                arrowprops=dict(arrowstyle="<->", color=fs.SECOND, lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(K + 0.9, np.sqrt(lam[K] * lam[K - 1]), f"gap ×{lam[K] / lam[K - 1]:.0f}", fontsize=6, color=fs.INK,
            va="center")
    ax.text(K + 0.9, lam[K - 1] / 6, "algebra, $k=4$", fontsize=6, color=fs.INK, ha="left", va="center")
    fs.tidy(ax)
    fig.text(0.585, 0.97, "(b)", fontsize=7, color=fs.INK, va="top", fontweight="bold")
    fig.text(0.81, 0.97, r"$Nc=\lambda Dc$ over the class, learned stepper", fontsize=6.5, color=fs.SECOND,
             va="top", ha="center")
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
