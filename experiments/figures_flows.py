"""Figure for the finite flows of exp45 (exp45_flows.py): equivariance defect of the finite transformation against its
size, one seed.

(a) polynomial class: the algebra and the six lowest eigenvectors (grey), the near-symmetries 7-10 (colour, solid),
the same with tau removed (colour, dashed), random fields (black); (b) directions of the trained m = 8 networks
(mu = 1 and mu = 0.01). Non-finite defects are drawn at the top edge (crosses).

  python figures_flows.py --dir results/neural2d/flows --seed 42 --out fig_flows.pdf
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = ["0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]
TOP, FLOOR = 3.0, 1e-13


def curve(g):
    x = np.array([float(k) for k in EPS])
    y = np.array([g["E"][k]["E"] for k in EPS], dtype=float)
    bad = ~np.isfinite(y) | (y > TOP)
    return x, np.clip(np.where(bad, TOP, y), FLOOR, TOP), bad


def draw(ax, g, **kw):
    x, y, bad = curve(g)
    ax.loglog(x, y, **kw)
    if bad.any():
        ax.loglog(x[bad], y[bad], ls="none", marker="x", ms=4, color=kw.get("color", "k"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="fig_flows.pdf")
    args = ap.parse_args()
    s = args.seed
    with open(os.path.join(args.dir, f"flows_s{s}.json")) as f:
        r = json.load(f)["gens"]
    nets = {}
    for tag in ("m8", "m8_mu001"):
        p = os.path.join(args.dir, f"flows_net_{tag}_s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                nets[tag] = json.load(f)["gens"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for n, g in r.items():
        if g["kind"] in ("true", "discovered"):
            draw(ax[0], g, color="0.6", lw=0.7, marker="o", ms=1.8)
    cols = {7: "C0", 8: "C1", 9: "C2", 10: "C3"}
    for i, c in cols.items():
        draw(ax[0], r[f"eigenvector {i}"], color=c, lw=1.1, marker="o", ms=2.5)
        draw(ax[0], r[f"eigenvector {i} without tau"], color=c, lw=0.9, ls="--", marker="s", ms=2)
    for j in (1, 2):
        draw(ax[0], r[f"random {j}"], color="k", lw=0.9, ls=":", marker="^", ms=2.5)
    ax[0].plot([], [], color="0.6", lw=0.7, marker="o", ms=1.8, label="algebra, eigenvectors 1-6")
    for i, c in cols.items():
        ax[0].plot([], [], color=c, lw=1.1, marker="o", ms=2.5, label=f"eigenvector {i}")
    ax[0].plot([], [], color="0.3", lw=0.9, ls="--", marker="s", ms=2, label=r"7-10, $\tau$ removed")
    ax[0].plot([], [], color="k", lw=0.9, ls=":", marker="^", ms=2.5, label=r"random ($\tau = 0$)")
    ax[0].set_title(f"(a) polynomial class, seed {s}")
    ax[0].set_ylabel(r"defect of the finite flow $E$")
    if nets:
        style = {"m8": dict(ls="-", marker="o"), "m8_mu001": dict(ls="--", marker="s")}
        for tag, gens in nets.items():
            for j in range(1, 9):
                g = gens[f"network direction {j}"]
                draw(ax[1], g, color="0.6" if j <= 6 else ("C4" if j == 7 else "C5"), lw=0.8 if j <= 6 else 1.1, ms=2,
                     **style[tag])
        ax[1].plot([], [], color="0.6", lw=0.8, marker="o", ms=2, label="directions 1-6")
        ax[1].plot([], [], color="C4", lw=1.1, marker="o", ms=2, label="direction 7")
        ax[1].plot([], [], color="C5", lw=1.1, marker="o", ms=2, label="direction 8")
        ax[1].plot([], [], color="0.3", lw=0.8, ls="-", label=r"$\mu = 1$")
        ax[1].plot([], [], color="0.3", lw=0.8, ls="--", label=r"$\mu = 0.01$")
        ax[1].set_title(f"(b) network directions ($m = 8$), seed {s}")
    ax[0].set_ylim(FLOOR / 2, TOP * 3)
    ax[1].set_ylim(3e-4, TOP * 3)
    for a, loc in zip(ax, ("lower right", "upper left")):
        a.set_xlabel(r"size of the transformation $\epsilon$ (nominal)")
        a.set_xlim(0.002, 1.5)
        leg = a.legend(frameon=True, framealpha=0.9, loc=loc, fontsize=5.5, handlelength=1.8)
        leg.get_frame().set_linewidth(0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
