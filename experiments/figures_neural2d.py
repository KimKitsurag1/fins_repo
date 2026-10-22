"""Figure for exp45: neural generators of u_t = div(u grad u) in 2D.

(a) held-out missing dimensions of the 6-dim algebra (characteristic space) during training, m = 6, one colour per
seed, exact operator solid and the learned operator (--overlay) dashed; grey band: Gaussian-coefficient 6-dim
subspaces of the polynomial class (mean +- sd). (b) eigenvalues of the transport pencil inside the learned span on
held-out pairs, m = 8 (trained through either operator, untrained), with the polynomial class's lowest
eigenvalues (training pairs) through either operator for reference.

  python figures_neural2d.py --dir $FINS_RESULTS_DIR/neural2d --prefix exact --overlay fno --out fig_neural2d.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FLOOR = 1e-9  # values below are drawn at the floor


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", default="exact")
    ap.add_argument("--overlay", default=None, help="prefix of learned-operator runs drawn dashed (m = 6)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="fig_neural2d.pdf")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    r6 = {s: load(os.path.join(args.dir, f"{args.prefix}_m6_s{s}.json")) for s in seeds}
    p8 = {s: os.path.join(args.dir, f"{args.prefix}_m8_s{s}.json") for s in seeds}
    r8 = {s: load(p) for s, p in p8.items() if os.path.exists(p)}
    po = {s: os.path.join(args.dir, f"{args.overlay}_m6_s{s}.json") for s in seeds} if args.overlay else {}
    ro = {s: load(p) for s, p in po.items() if os.path.exists(p)}
    po8 = {s: os.path.join(args.dir, f"{args.overlay}_m8_s{s}.json") for s in seeds} if args.overlay else {}
    ro8 = {s: load(p) for s, p in po8.items() if os.path.exists(p)}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    colors = ["C0", "C1", "C2", "C3"]
    nulls = np.array([[r["random_null"]["missing_q"], r["random_null"]["std_missing_q"]] for r in r6.values()])
    steps = [c["step"] for c in next(iter(r6.values()))["curve"]]
    mu, sd = nulls[:, 0].mean(), nulls[:, 1].mean()
    ax[0].axhspan(mu - sd, mu + sd, color="0.85", lw=0)
    ax[0].axhline(mu, color="0.55", lw=0.8, label="random 6-dim subspaces")
    for c, (s, r) in zip(colors, r6.items()):
        ax[0].plot([q["step"] for q in r["curve"]], [q["missing_q"] for q in r["curve"]], color=c, lw=1.1,
                   marker="o", ms=2.5, label=f"seed {s}")
        if s in ro:
            ax[0].plot([q["step"] for q in ro[s]["curve"]], [q["missing_q"] for q in ro[s]["curve"]], color=c,
                       lw=1.0, ls="--", marker="^", ms=2.5)
    if ro:
        ax[0].plot([], [], color="0.2", lw=1.1, marker="o", ms=2.5, label="exact operator")
        ax[0].plot([], [], color="0.2", lw=1.0, ls="--", marker="^", ms=2.5, label="learned FNO2d")
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.05, 5)
    ax[0].set_xlim(0, max(steps))
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("missing dimensions (held-out)")
    ax[0].set_title("(a) neural generators, m = 6")
    ax[0].legend(frameon=False, loc="center right", fontsize=6)

    for c, (s, r) in zip(colors, r8.items()):
        k = np.arange(1, len(r["span_eigs_trained"]) + 1)
        ax[1].semilogy(k, np.clip(r["span_eigs_trained"], FLOOR, None), color=c, marker="o", ms=3, lw=0.9)
        ax[1].semilogy(k, np.clip(r["span_eigs_untrained"], FLOOR, None), color=c, marker="x", ms=3, lw=0.6, ls=":")
        if s in ro8:
            ax[1].semilogy(k, np.clip(ro8[s]["span_eigs_trained"], FLOOR, None), color=c, marker="^", ms=3, lw=0.8,
                           ls="--")
    if r8:
        r = next(iter(r8.values()))
        s0 = next(iter(r8))
        lam = np.clip(np.array(r["poly_lam"][:10]), FLOOR, None)
        ax[1].semilogy(np.arange(1, len(lam) + 1), lam, color="0.3", marker="s", ms=2.5, lw=0.7, ls="-.")
        if s0 in ro:
            lo = np.clip(np.array(ro[s0]["poly_lam"][:10]), FLOOR, None)
            ax[1].semilogy(np.arange(1, len(lo) + 1), lo, color="0.6", marker="s", ms=2.5, lw=0.7, ls="-.")
        ax[1].axvline(6.5, color="0.7", lw=0.6)
        ax[1].plot([], [], color="0.2", marker="o", ms=3, lw=0.9, label="trained, exact")
        if ro8:
            ax[1].plot([], [], color="0.2", marker="^", ms=3, lw=0.8, ls="--", label="trained, FNO2d")
        ax[1].plot([], [], color="0.2", marker="x", ms=3, lw=0.6, ls=":", label="untrained")
        ax[1].plot([], [], color="0.3", marker="s", ms=2.5, lw=0.7, ls="-.", label=f"polynomial, exact, {s0}")
        if s0 in ro:
            ax[1].plot([], [], color="0.6", marker="s", ms=2.5, lw=0.7, ls="-.", label=f"polynomial, FNO2d, {s0}")
    ax[1].set_ylim(FLOOR / 3, 3)
    ax[1].set_xlabel("index")
    ax[1].set_ylabel("eigenvalue")
    ax[1].set_title("(b) spectrum inside the learned span, m = 8")
    ax[1].legend(frameon=False, loc="lower right", ncol=1, fontsize=5.5, handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
