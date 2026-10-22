"""Experiment 59 -- transport by a nonlocal velocity: the shift in place of the boost, read without the equation
(README_EXPERIMENTS.md, pre-registration of 08.10).

u_t + (H u) u_x = -nu Lambda^alpha u on the torus (ccf.py), alpha in {1.25, 1.5, 1.75}; algebra d_x, d_t, the shift d_u
(no Galilean boost) and x d_x + alpha t d_t + (1 - alpha) u d_u. The transport term is bilinear and nonlocal: outside the
local library and outside its linear nonlocal completion. Conditions clean and 1e-2, seeds 42-51, data seed 1000 + s:
  weak          FINS weak through the exact solver (exp49 with exp57's settings)
  wsindy        reference weak SINDy then Lie (exp56_wsindy.py), evaluation data
  regress       regression rows on the evaluation data, one json per run (this script's 'one'):
                  ref       the reference library d_x^k (u^j), j <= 4, k <= 6, and a constant
                  prof-free ref plus E_beta, O_beta (symbols -|k|^beta, i sgn(k) |k|^beta), beta minimising the selected
                            model's residual (exp58)
                  bil-free  prof-free plus bilinear nonlocal columns d_x(u Hu), d_x((Hu)^2), u Lambda u, Lambda(u^2), which
                            span (Hu) u_x = d_x(u Hu) - u Lambda u, u (Hu)_x, H(u u_x), ((Hu)^2)_x
                  nested    ref, replaced by prof-free and then by bil-free only when the weak residual drops by more
                            than NEST times
  weak-learned  FINS weak through FNO steppers trained on 1024 clean 'mixed' trajectories (notebook 7), seeds 42-44
Readouts: FINS's span (exp58.readout_file): the boost plane's deepest direction (the shift when its t d_x share is below
SHARE, a boost above) and the time-scaling exponent a-hat; a regression row: the algebra of its identified equation
(shift, boost and the exponent at which its terms balance), and whether it is the true equation.

Run:  python exp59_ccf.py run [--pdes ccf150,...] [--conds clean,n1e-2] [--seeds ...] [--parts ...]
      python exp59_ccf.py one --pde ccf150 --cond n1e-2 --seed 42 --noise 0.01 --out <json>
      python exp59_ccf.py summary
"""

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import ccf                                                 # noqa: E402
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp50_two_stage as e50                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp56_wsindy as e56                                 # noqa: E402
import exp57_fburgers as e57                               # noqa: E402
import exp58_symmetry_library as e58                       # noqa: E402
import method_wsindy as mw                                 # noqa: E402

CONDS = {"clean": 0.0, "n1e-2": 1e-2}
PARTS = "weak,wsindy,regress,weak-learned"
NEST = 1.25
EXP_TOL = 0.05           # |a - alpha| for the right scaling
CANCEL = 0.1             # relative tolerance of the Lambda u cancellation in the shift test
DEG_TOL = e58.DEG_TOL
BIL = (("P1", 1), ("P2", 1), ("P3", 0), ("P4", 0))   # d_x(u Hu), d_x((Hu)^2), u Lambda u, Lambda(u^2)


# ---------------------------------------------------------------------------------------------------------------------
# the regression rows

class Lib(e58.Lib):
    """exp58's library (reference columns, multipliers at any beta) plus the bilinear nonlocal columns; MSTLS with the
    library's own size as J."""

    def __init__(self, U, x, t):
        super().__init__(U, x, t)
        p = self.w.prep
        kap = 2 * np.pi * np.fft.fftfreq(p["Nx"], d=p["dx"])
        hil, lam = -1j * np.sign(kap), np.abs(kap) * p["sx"] ** (-1)

        def op(sym, f):
            return np.real(np.fft.ifft(sym[:, None] * np.fft.fft(f, axis=0), axis=0))

        Gs = []
        for u in U:
            us = u / p["su"]
            hu = op(hil, us)
            fields = (us * hu, hu ** 2, us * op(lam, us), op(lam, us ** 2))
            Gs.append(np.stack([mw.conv_valid(f, (p["Cx"][k], p["Ct"][0]), (p["s_x"], p["s_t"])).ravel()
                                for f, (_, k) in zip(fields, BIL)], 1))
        Gb = np.concatenate(Gs)
        Mb = np.array([p["su"] ** -2 * p["sx"] ** -(k + (1 if name in ("P3", "P4") else 0)) for name, k in BIL])
        for i, key in enumerate(BIL):
            self.cols[key] = (Gb[:, i], Mb[i])
        self.local = self.local + list(BIL)
        self.G = np.concatenate([self.G, Gb], 1)
        self.Mloc = np.concatenate([self.Mloc, Mb])
        self.A, self.c = self.G.T @ self.G, self.G.T @ self.b
        self.reference = self.local[:-len(BIL)]

    def fit(self, local, beta=None, nterms=None, tol=1e-9):
        return super().fit(local, beta, nterms=nterms, tol=tol)

    def resid_gram(self, local, beta, tol=1e-9):
        idx = [self.local.index(jk) for jk in local]
        m = self.multipliers(beta)
        mi = [{"E": 0, "O": 1}[kd] for kd in self.kinds(local, beta, tol)]
        A = np.block([[self.A[np.ix_(idx, idx)], m["X"][np.ix_(idx, mi)]],
                      [m["X"][np.ix_(idx, mi)].T, m["A"][np.ix_(mi, mi)]]])
        c = np.concatenate([self.c[idx], m["c"][mi]])
        M = np.concatenate([self.Mloc[idx], m["M"][mi]]) / self.Mlhs
        return e58.mstls_gram(A, c, self.bb, M, nterms=len(c))


def kind_of(t):
    return t["j"] if isinstance(t["j"], str) else t["kind"]


def name_of(terms):
    out = []
    for t in terms:
        k = kind_of(t)
        if k == "loc":
            out.append(f"{t['c']:+.5g} d{t['k']}(u^{t['j']})")
        elif k in ("E", "O"):
            out.append(f"{t['c']:+.5g} {k}^{t['beta']:.3f} u")
        else:
            out.append(f"{t['c']:+.5g} {dict(P1='(u Hu)_x', P2='((Hu)^2)_x', P3='u Lu', P4='L(u^2)')[k]}")
    return " ".join(out)


def algebra(terms):
    """(shift, boost, advection coefficient k of a boost t d_x + d_u / k, exponent a at which the terms balance) of
    u_t = sum of terms. Under u -> u + eps: d_x^k (u^j) varies by j d_x^k (u^(j - 1)); d_x(u Hu) and u Lambda u by
    Lambda u, Lambda(u^2) by 2 Lambda u; E, O, d_x((Hu)^2), the constant do not vary."""
    var, lam_c, lam_n = {}, 0.0, 0.0
    orders = []
    for t in terms:
        k = kind_of(t)
        if k == "loc":
            j, kk = t["j"], t["k"]
            orders.append((j, kk))
            if j >= 1 and not (j == 1 and kk >= 1):
                var[(j - 1, kk)] = var.get((j - 1, kk), 0.0) + j * t["c"]
        elif k in ("E", "O"):
            orders.append((1, t["beta"]))
        else:
            orders.append((2, 1))
            w = {"P1": 1.0, "P3": 1.0, "P4": 2.0}.get(k, 0.0)
            lam_c += w * t["c"]
            lam_n += abs(w * t["c"])
    lam_ok = abs(lam_c) <= CANCEL * lam_n if lam_n else True
    shift = not var and lam_ok
    adv = var.pop((1, 1), 0.0) if set(var) <= {(1, 1)} else 0.0          # (u^2)_x varies by 2 u_x
    boost = adv != 0 and not var and lam_ok
    a = e56.balance(orders) if orders else None
    return dict(shift=bool(shift), boost=bool(boost), k=float(-adv) if boost else None, a=a)


def score(pde, terms, x, u_clean):
    """The true equation (u_t = -(u Hu)_x + u Lambda u + nu E_alpha u: these terms and no others, the order within
    DEG_TOL), the algebra and its correctness, the relative right-hand-side error on the clean data."""
    al = ccf.alpha(pde)
    kinds = sorted(kind_of(t) for t in terms)
    lin = [t for t in terms if kind_of(t) in ("E", "O") or (kind_of(t) == "loc" and t["j"] == 1 and t["k"] >= 1)]
    ok = kinds == ["E", "P1", "P3"] and abs(lin[0]["beta"] - al) <= DEG_TOL
    alg = algebra(terms)
    right = alg["shift"] and not alg["boost"] and alg["a"] is not None and abs(alg["a"] - al) <= EXP_TOL
    return dict(true_eq=bool(ok), right_algebra=bool(right), eF=rhs_error(pde, terms, x, u_clean), **alg)


def rhs(terms, x, u):
    kap = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])

    def op(sym, f):
        return np.real(np.fft.ifft(sym[None, :, None] * np.fft.fft(f, axis=1), axis=1))

    hu, lu = op(-1j * np.sign(kap), u), op(np.abs(kap), u)
    out = np.zeros_like(u)
    for t in terms:
        k = kind_of(t)
        if k == "loc":
            f = u ** t["j"] if t["j"] else np.ones_like(u)
            out += t["c"] * (op((1j * kap) ** t["k"], f) if t["k"] else f)
        elif k in ("E", "O"):
            out += t["c"] * op(e58.symbol("even" if k == "E" else "odd", t["beta"], kap), u)
        else:
            f = {"P1": op(1j * kap, u * hu), "P2": op(1j * kap, hu ** 2), "P3": u * lu,
                 "P4": op(np.abs(kap), u ** 2)}[k]
            out += t["c"] * f
    return out


def rhs_error(pde, terms, x, u):
    al = ccf.alpha(pde)
    true = [dict(kind="loc", j="P1", k=1, beta=None, c=-1.0), dict(kind="loc", j="P3", k=0, beta=None, c=1.0),
            dict(kind="E", j=1, k=0, beta=al, c=ccf.nu(al))]
    F0 = rhs(true, x, u)
    return float(np.linalg.norm(rhs(terms, x, u) - F0) / np.linalg.norm(F0))


def one(args):
    cfg = e31.ExperimentConfig()
    a = argparse.Namespace(pde=args.pde, n=10, dseed=args.data_seed or 1000 + args.seed, noise=args.noise, thin=1)
    t0 = time.time()
    u, u_clean, xs, ts, _ = e50.data(a, cfg)
    x, t = xs[0], ts[0]
    lib = Lib(u, x, t)
    print(f"[{args.pde}] {args.cond}, seed {args.seed} (data {a.dseed}): data and columns {time.time() - t0:.0f} s; "
          f"{lib.w.info['rows']} rows", flush=True)
    rows = {}

    def put(name, terms, resid, lam, beta=None):
        rows[name] = dict(terms=terms, names=name_of(terms), resid=resid, lam=lam, beta=beta,
                          **score(args.pde, terms, x, u_clean))
        r = rows[name]
        print(f"  {name:10s} {'TRUE ' if r['true_eq'] else 'wrong'} algebra {'right' if r['right_algebra'] else 'WRONG'}"
              f" (shift {r['shift']}, boost {r['boost']}, a {r['a']}) eF {r['eF']:.3g}"
              + (f" beta {beta:.3f}" if beta is not None else "") + f"  u_t = {r['names']}", flush=True)

    put("ref", *lib.fit(lib.reference))
    terms, resid, lam, b = lib.profile(lambda bb: lib.reference, e58.BETAS)
    put("prof-free", terms, resid, lam, beta=b)
    terms, resid, lam, b = lib.profile(lambda bb: lib.local, e58.BETAS)
    put("bil-free", terms, resid, lam, beta=b)
    pick = "ref"
    for nxt in ("prof-free", "bil-free"):
        if rows[pick]["resid"] > NEST * rows[nxt]["resid"]:
            pick = nxt
    rows["nested"] = dict(rows[pick], picked=pick)
    print(f"  nested     -> {pick}; total {time.time() - t0:.0f} s", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(pde=args.pde, cond=args.cond, seed=args.seed, data_seed=a.dseed, noise=args.noise,
                           rows=rows, wsindy={k: float(v) for k, v in lib.w.info.items()}), fh, indent=1)
        print(f"  saved {args.out}")


# ---------------------------------------------------------------------------------------------------------------------
# jobs, readouts and summary

def jobs(pdes, conds, seeds, parts, runs):
    out = []
    for c in conds:
        d = os.path.join(runs, c)
        nz = ["--noise", f"{CONDS[c]:g}"]
        for pde in pdes:
            for seed in seeds:
                ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
                p = ["--pde", pde]
                if "weak" in parts:
                    f = os.path.join(d, f"weak_{pde}_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", *p, "--op", "exact", *ds, *nz, *e57.WEAK, "--save", f]))
                if "wsindy" in parts:
                    pre = os.path.join(d, f"wsindy_{pde}_s{seed}")
                    out.append((pre + "_soft.npz", ["exp56_wsindy.py", *p, *ds, "--data", "eval", *nz, "--out", pre]))
                if "regress" in parts:
                    f = os.path.join(d, f"regress_{pde}_s{seed}.json")
                    out.append((f, ["exp59_ccf.py", "one", *p, "--cond", c, *ds, *nz, "--out", f]))
                if "weak-learned" in parts and seed in e51.FIT_SEEDS and c == "clean":
                    f = os.path.join(d, f"weak-learned_{pde}_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", *p, "--op", "learned", *ds, "--tag", "", *nz, *e57.WEAK,
                                    "--save", f]))
    return out


def fins_readouts(runs, pdes, conds, seeds):
    """{cond: {pde: {row: {seed: readout}}}} of the FINS weak files (exp58.readout_file), rows 'FINS weak' (own count),
    'FINS weak, m = k' and the learned operator's."""
    res = {}
    for c in conds:
        for pde in pdes:
            for s in seeds:
                for tag, lab in (("weak", "FINS weak"), ("weak-learned", "FINS weak, learned operator")):
                    p = os.path.join(runs, c, f"{tag}_{pde}_s{s}.npz")
                    if not os.path.exists(p):
                        continue
                    r = e58.readout_file(p, pde)
                    res.setdefault(c, {}).setdefault(pde, {}).setdefault(lab, {})[str(s)] = r["own"]
                    res[c][pde].setdefault(lab + ", m = k", {})[str(s)] = r["mk"]
    return res


def fins_algebra(r, al):
    """(shift, boost, a-hat) of a readout: the shift when the boost plane's deepest direction is contained with t d_x
    share below SHARE."""
    contained = r["lam_boost"] >= e58.GATE
    shift = contained and r["share"] < e58.SHARE
    boost = contained and r["share"] >= e58.SHARE
    a = r["a"] if r["c"] >= e58.GATE else None
    right = shift and a is not None and abs(a - al) <= EXP_TOL
    return dict(shift=bool(shift), boost=bool(boost), a=a, right_algebra=bool(right))


def summary(args):
    pdes, conds, seeds = args.pdes.split(","), args.conds.split(","), args.seeds.split(",")
    fr = fins_readouts(args.runs, pdes, conds, seeds)
    res, lines = {}, ["# Transport by a nonlocal velocity (exp59): the right algebra (shift, no boost, |a - alpha| <= "
                      f"{EXP_TOL}) and the true equation, counts over seeds; median a-hat\n"]
    for c in conds:
        for pde in pdes:
            al = ccf.alpha(pde)
            cell = res.setdefault(c, {}).setdefault(pde, {})
            for lab, v in fr.get(c, {}).get(pde, {}).items():
                cell[lab] = {s: dict(fins_algebra(r, al), readout=r) for s, r in v.items()}
            js = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.runs, c, f"regress_{pde}_s*.json")))]
            for row in ("ref", "prof-free", "bil-free", "nested"):
                cell[row] = {str(j["seed"]): j["rows"][row] for j in js if row in j["rows"]}
    for c in conds:
        lines.append(f"## {c}\n")
        lines.append("| row | " + " | ".join(pdes) + " |")
        lines.append("|---" * (len(pdes) + 1) + "|")
        labels = []
        for pde in pdes:
            labels += [lab for lab in res.get(c, {}).get(pde, {}) if lab not in labels]
        for lab in labels:
            cells = []
            for pde in pdes:
                v = res[c].get(pde, {}).get(lab, {})
                if not v:
                    cells.append("")
                    continue
                n = len(v)
                ra = sum(r["right_algebra"] for r in v.values())
                sh = sum(r["shift"] for r in v.values())
                bo = sum(r["boost"] for r in v.values())
                aa = [r["a"] for r in v.values() if r["a"] is not None]
                te = sum(r.get("true_eq", False) for r in v.values())
                cells.append(f"right {ra}/{n}, shift {sh}, boost {bo}" + (f", true {te}" if "true_eq" in
                                                                                     next(iter(v.values())) else "")
                             + (f", a {np.median(aa):.3f}" if aa else ", a none"))
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "one", "summary"])
    ap.add_argument("--pde", default=None)
    ap.add_argument("--cond", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--pdes", default=",".join(ccf.EQUATIONS))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--parts", default=PARTS)
    base = os.environ.get("FINS_RESULTS_DIR", "results")
    ap.add_argument("--runs", default=os.path.join(base, "ccf_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.what == "one":
        one(args)
    elif args.what == "run":
        e51.run(args, jobs(args.pdes.split(","), args.conds.split(","), [int(s) for s in args.seeds.split(",")],
                           args.parts.split(","), args.runs))
    else:
        args.out = args.out or os.path.join(e51.HERE, "results", "ccf")
        summary(args)


if __name__ == "__main__":
    main()
