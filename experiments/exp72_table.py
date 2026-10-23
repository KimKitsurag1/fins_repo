"""Rows of the paper's table tab:use (exp72: a discovered symmetry as data augmentation) and the numbers the text of its
section quotes for the reported runs, from results/utility (summary.json, spans_<pde>_s<seed>.npz). --zero-mean adds
the boost that FINS reads through the zero-mean Kuramoto-Sivashinsky steppers (exp53's ablation:
weak-learned-attractor_ks_s<seed>.npz under FINS_RESULTS_DIR/ks_runs/clean); --spans prints the boost of further spans
files of KS (e.g. the development run's closed-loop span, fins_runs/utility_dev/ks_mixed_n256_e150).
Speed ratio of a boost element xi(t) d_x + phi(t) d_u with xi(0) = 0, phi(0) = PHI0: xi'(t) PHI0 / phi(t), 1 for the
boost; reported at t = 0, at t = T and as its range over [0, T].

Run:  python exp72_table.py [--zero-mean] [--spans FILE ...]
"""

import argparse
import json
import os

import numpy as np

import exp72_utility as e72

ROWS = (("no augmentation", "none"), ("span of FINS", "fins"), ("span of LieNLSD", "lienlsd"),
        ("true algebra", "oracle"), ("constant without drift", "shift"), ("real backgrounds within $\\pm3$", "wide"))
D = os.path.join(e72.HERE, "results", "utility")


def fmt(v):
    """Two significant digits, as the table prints them."""
    return f"{v:.{max(0, 1 - int(np.floor(np.log10(abs(v)))))}f}"


def rng(v):
    return f"{min(v):.3g}-{max(v):.3g}"


def speed(be, pde, T):
    """Speed ratio of a boost element at t = 0, at t = T, and its minimum and maximum over [0, T]."""
    p0, t = e72.PHI0[pde], np.linspace(0.0, T, 1001)
    r = (be["a1"] + 2 * be["a2"] * t) * p0 / (p0 + be["b1"] * t + be["b2"] * t ** 2)
    return r[0], r[-1], r.min(), r.max()


def speed_text(sp):
    return f"speed {sp[0]:.3f} at 0, {sp[1]:.3f} at T, {sp[2]:.3f}-{sp[3]:.3f} over [0, T]"


def part(f, arm, pde):
    """Dimension of the part in W, its largest principal angle to the true part (rad) and the missing dimensions of the
    true part, and the boost element's speed ratios at 0 and T."""
    V, pc, be = f[f"V_{arm}"], f[f"to_oracle_{arm}"], json.loads(str(f[f"boost_{arm}"]))
    ang = float(np.arccos(np.clip(pc.min(), -1, 1))) if len(pc) == 2 else float("nan")
    return V.shape[1], ang, 2 - float(np.sum(pc ** 2)), (speed(be, pde, float(f["T_max"])) if be else None)


def zero_mean():
    """The boost element of FINS's part in W through the zero-mean KS steppers."""
    R = os.environ.get("FINS_RESULTS_DIR", "results")
    for seed in e72.SEEDS:
        p = os.path.join(R, "ks_runs", "clean", f"weak-learned-attractor_ks_s{seed}.npz")
        if not os.path.exists(p):
            print(f"% zero-mean KS s{seed}: {p} missing")
            continue
        z = e72.reference("ks", seed)
        G = e72.metric(e72.class_vectors(z))
        f = np.load(p, allow_pickle=True)
        e72.check_reference(f, z, p)
        V, cos = e72.torus_part(f["C"][:, :int(f["nown"])], z, G)
        sp = speed(e72.boost_element(V, z, "ks"), "ks", z["T_max"]) if V.shape[1] >= 2 else None
        Qo = e72.g_orth(e72.torus_part(e72.oracle_span("ks", z), z, G)[0], G)
        pc = np.linalg.svd(Qo.T @ G @ V, compute_uv=False)
        print(f"% zero-mean KS s{seed}: own count {int(f['nown'])}, part in W {V.shape[1]} (cosines "
              + " ".join(f"{c:.4f}" for c in cos) + "; to the true part " + " ".join(f"{c:.6f}" for c in pc) + ")"
              + (", " + speed_text(sp) if sp else ", no boost element"))


def extra(paths):
    """The boost element of the part in W of further KS spans files (key V_fins)."""
    for p in paths:
        f = np.load(p, allow_pickle=True)
        z = dict(mons=f["mons"], T_max=float(f["T_max"]))
        be = e72.boost_element(f["V_fins"], z, "ks")
        print(f"% {p}: xi = {be['a1']:.4f} t {be['a2']:+.2e} t^2, phi = 1 {be['b1']:+.4f} t {be['b2']:+.2e} t^2 "
              f"(phi(T) - phi(0) = {be['b1'] * z['T_max'] + be['b2'] * z['T_max'] ** 2:+.3f}), "
              + speed_text(speed(be, "ks", z["T_max"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zero-mean", action="store_true")
    ap.add_argument("--spans", nargs="*", default=[])
    args = ap.parse_args()
    err = json.load(open(os.path.join(D, "summary.json")))["errors"]

    def e(pde, seed, arm, where):
        return err[f"{pde} {seed}"][arm]["in" if where == "ID" else "out"][1]

    print("% tab:use: 16-step error in percent, median over seeds 42-44; KS, KdV, BO, each ID / shifted")
    for name, arm in ROWS:
        if arm == "wide":
            print("\\midrule")
        cells = [fmt(100 * np.median([e(p, s, arm, w) for s in e72.SEEDS])) for p in e72.EQS for w in ("ID", "OOD")]
        print(f"{name} & " + " & ".join(cells) + " \\\\")
    runs = [(p, s) for p in e72.EQS for s in e72.SEEDS]
    spans = {(p, s): np.load(e72.spans_file(p, s), allow_pickle=True) for p, s in runs}
    with_part = [(p, s) for p, s in runs if spans[(p, s)]["V_fins"].shape[1]]
    print("% none: shifted / ID " + rng([e(p, s, "none", "OOD") / e(p, s, "none", "ID") for p, s in runs]))
    for p in e72.EQS:
        r = [(s, e(p, s, "none", "OOD") / e(p, s, "fins", "OOD")) for q, s in with_part if q == p]
        print(f"% {p}: none / fins, shifted (seeds with a part) " + ", ".join(f"{s} {v:.3g}" for s, v in r))
    for ref in ("oracle", "wide"):
        print(f"% fins / {ref}, shifted: " + ", ".join(f"{p} {s} {e(p, s, 'fins', 'OOD') / e(p, s, ref, 'OOD'):.3g}"
                                                       for p, s in with_part))
    print("% shift / none, shifted " + rng([e(p, s, "shift", "OOD") / e(p, s, "none", "OOD") for p, s in runs]))
    for arm in ("fins", "lienlsd", "oracle", "wide"):
        print(f"% {arm} / none, ID: max {max(e(p, s, arm, 'ID') / e(p, s, 'none', 'ID') for p, s in runs):.3g}")
    print("% lienlsd / oracle, shifted, KS: " + ", ".join(
        f"{s} {e('ks', s, 'lienlsd', 'OOD') / e('ks', s, 'oracle', 'OOD'):.3g}" for s in e72.SEEDS))
    print("% parts in W: dimension, largest principal angle to the true part (rad), its missing dimensions, speed "
          "ratio of the boost element at 0 and T")
    for p, s in runs:
        for arm in ("fins", "lienlsd"):
            m, ang, miss, sp = part(spans[(p, s)], arm, p)
            print(f"% {p} {s} {arm}: own count {int(spans[(p, s)][f'own_{arm}'])}, part {m}, angle {ang:.4f}, "
                  f"missing {miss:.2e}" + (", " + speed_text(sp) if sp else ""))
    if args.zero_mean:
        zero_mean()
    extra(args.spans)


if __name__ == "__main__":
    main()
