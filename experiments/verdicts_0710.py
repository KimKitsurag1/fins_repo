"""Verdicts of the predictions registered on 06.10-07.10 (README_EXPERIMENTS.md): WN1-WN10 (exp51), BO1-BO6 (exp52),
KS1-KS5 (exp53), NL1-NL9 (learned arms, exp54, LieNLSD on KS), EX1-EX4 (exp55, notebook 4), LW1-LW5 (exp56, notebook 5), FB1-FB5
(exp57, notebook 6), SL1-SL6 (exp58), AN1-AN5 and AR1-AR2 (exp60, notebook 7), PG1-PG8 and WS1-WS5 (exp61, notebook 8),
DT1-DT5 (exp62), UV1-UV5 (exp63), QL1-QL4 (exp64), EA1-EA3 (exp65, notebook 9), LS1-LS2 (exp66, notebook 9), CT1-CT2
(exp67), EP0-EP5 (exp68, notebook 10), TR0-TR8 (exp69, notebook 11), read from the summaries' JSON and the exponent readouts (exp56_tables.py readouts, exp57, exp58, exp60, exp61
and exp62 summaries).
NOT MET if any part fails, else PENDING if any part lacks seeds, else MET.

Run (after the summaries):  python verdicts_0710.py [--results results]
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

SEEDS = [str(s) for s in range(42, 52)]
FIT = ["42", "43", "44"]
TS = ("two-stage, H 0.2, soft", "two-stage, H 0.2, hard (own count)")
TS05_HARD = "two-stage, H 0.05, hard (own count)"
LEARNED = ("FINS weak, learned operator", "FINS pointwise, learned operator")
ZERO_MEAN = ("FINS weak, learned on zero-mean data", "FINS pointwise, learned on zero-mean data")
WS_EVAL = ("WSINDy, evaluation data, soft", "WSINDy, evaluation data, hard (own count)")
WS_TRAIN = ("WSINDy, training data, soft", "WSINDy, training data, hard (own count)")
HEAT_KDV = ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1")
KS_CONDS = ("clean", "n1e-2", "n3e-2", "n5e-2", "n1e-1")
FB2_TOL = 0.15
AN_EXPONENTS, AN_EPS = ("1.25", "1.5", "1.75"), 0.6


class Table:
    """{seed: value} of a summary row: (condition, equation, row, algebra, key)."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, c, pde, row, cv="finite", key="missing"):
        return self.d.get(c, {}).get(pde, {}).get(row, {}).get(cv, {}).get(key, {})


class Exponent(Table):
    """{seed: [a-hat, containment]} of a row: (condition, equation, row)."""

    def __call__(self, c, pde, row):
        return self.d.get(c, {}).get(pde, {}).get(row, {})


class Check:
    def __init__(self, name):
        self.name, self.fail, self.pending, self.obs = name, False, False, []

    def has(self, *vs, seeds=SEEDS):
        ok = all(all(s in v for s in seeds) for v in vs)
        self.pending |= not ok
        return ok

    def test(self, cond, text):
        self.fail |= not cond
        self.obs.append(text)

    def group(self, cells, rule, need, text, seeds=SEEDS):
        """One observation for many cells [(label, {seed: value})]: each needs `need` seeds passing; the worst shown."""
        if not self.has(*[v for _, v in cells], seeds=seeds):
            return
        ns = [(count(v, seeds, rule), lab) for lab, v in cells]
        n, lab = min(ns)
        self.test(all(m >= need for m, _ in ns), f"{text}: {len(cells)} cells, worst {n}/{len(seeds)} ({lab})")

    def line(self):
        tag = "NOT MET" if self.fail else ("PENDING" if self.pending else "MET")
        return f"({self.name}) {tag}: " + "; ".join(self.obs)


def mean(v, seeds):
    return float(np.mean([v[s] for s in seeds]))


def count(v, seeds, ok):
    return sum(bool(ok(v[s])) for s in seeds)


def seed_rule(C, v, rule, need, text, seeds=SEEDS):
    if C.has(v, seeds=seeds):
        n = count(v, seeds, rule)
        C.test(n >= need, f"{text} {n}/{len(seeds)}")


def mean_rule(C, v, rule, text, seeds=SEEDS):
    if C.has(v, seeds=seeds):
        m = mean(v, seeds)
        C.test(rule(m), f"{text} {m:.3f}")


def wn(T):
    out, eqs, w, p = [], ("heat", "kdv"), "FINS weak", "FINS pointwise"
    C = Check("WN1")
    for pde in eqs:
        for c in ("clean", "n1e-3"):
            seed_rule(C, T(c, pde, w), lambda x: x <= 0.05, 9, f"{pde} {c}")
    out.append(C)
    for name, conds, rule in (("WN2", ("n1e-3", "n1e-2"), lambda a, b: a <= b / 10),
                              ("WN3", ("n3e-2", "n5e-2", "n1e-1"), lambda a, b: a < b)):
        C = Check(name)
        for pde in eqs:
            for c in conds:
                a, b = T(c, pde, w), T(c, pde, p)
                if C.has(a, b):
                    C.test(rule(mean(a, SEEDS), mean(b, SEEDS)),
                           f"{pde} {c} weak {mean(a, SEEDS):.3f} vs pointwise {mean(b, SEEDS):.3f}")
        out.append(C)
    C = Check("WN4")
    for pde, c, bound in (("heat", "n1e-2", 0.05), ("heat", "n5e-2", 0.3), ("kdv", "n1e-2", 0.1), ("kdv", "n5e-2", 1.0)):
        mean_rule(C, T(c, pde, w), lambda m, b=bound: m <= b, f"{pde} {c} (<= {bound})")
    out.append(C)
    C = Check("WN5")
    for c in ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2"):
        mean_rule(C, T(c, "kdv", w, "non-trivial"), lambda m: m <= 0.05, f"KdV non-trivial {c}")
    out.append(C)
    C = Check("WN6")
    for pde in eqs:
        for c in ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1"):
            for row in TS:
                seed_rule(C, T(c, pde, row), lambda x: x <= 0.05, 9, f"{pde} {c} {row[11:]}")
    out.append(C)
    C = Check("WN7")
    for pde in eqs:
        for c in ("n5e-2", "n1e-1"):
            seed_rule(C, T(c, pde, TS05_HARD), lambda x: x >= 1, 5, f"{pde} {c}")
    out.append(C)
    C = Check("WN8")
    for pde in eqs:
        for c in ("n1e-3", "n1e-2", "n3e-2", "n5e-2"):
            a, b = T(c, pde, "LieNLSD_RHS@eval"), T(c, pde, w)
            if C.has(a, b):
                C.test(mean(a, SEEDS) > mean(b, SEEDS),
                       f"{pde} {c} LieNLSD {mean(a, SEEDS):.3f} vs weak {mean(b, SEEDS):.3f}")
    out.append(C)
    C = Check("WN9")
    for pde, k, cv in (("heat", 10, "full"), ("kdv", 4, "finite")):
        for c in ("clean", "n1e-3", "n1e-2"):
            seed_rule(C, T(c, pde, w + ", own count", cv, "k"), lambda x, k=k: round(x) == k, 8, f"{pde} {c} own = {k}")
    out.append(C)
    C = Check("WN10")
    for c, bound in (("clean", 0.05), ("n1e-3", 0.3), ("n1e-2", 0.3)):
        ref = {"n1e-3": 1.18, "n1e-2": 2.20}.get(c)
        lim = bound if ref is None else min(bound, ref / 5)
        mean_rule(C, T(c, "kdv", LEARNED[0]), lambda m, b=lim: m <= b, f"KdV learned weak {c} (<= {lim:.3f})", FIT)
    out.append(C)
    return out


def bo(T):
    out, conds = [], ("clean", "n1e-2", "n3e-2")
    C = Check("BO1")
    for row in ("FINS weak", "FINS pointwise"):
        seed_rule(C, T("clean", "bo", row), lambda x: x <= 0.05, 9, row)
    out.append(C)
    C = Check("BO2")
    for H in ("H 0.2", "H 0.05"):
        for c in conds:
            for cv in ("finite", "non-trivial"):
                seed_rule(C, T(c, "bo", f"two-stage, {H}, hard (own count)", cv), lambda x: x >= 1.5, 9,
                          f"{H} {c} {cv}")
    out.append(C)
    C = Check("BO3")
    for c in conds:
        wv = T(c, "bo", "FINS weak", "non-trivial")
        if not C.has(wv):
            continue
        mw = mean(wv, SEEDS)
        C.test(mw <= 0.05, f"{c} weak {mw:.3f} (<= 0.05)")
        for H in ("H 0.2", "H 0.05"):
            h = T(c, "bo", f"two-stage, {H}, hard (own count)", "non-trivial")
            if C.has(h):
                C.test(mw < mean(h, SEEDS), f"{c} {H} hard {mean(h, SEEDS):.3f}")
        if c in ("clean", "n1e-2"):
            for row in ("LieNLSD_F", "LieNLSD_F@eval"):
                f = T(c, "bo", row, "non-trivial")
                if C.has(f, seeds=FIT):
                    C.test(mean(wv, FIT) < mean(f, FIT), f"{c} {row} {mean(f, FIT):.3f} vs weak {mean(wv, FIT):.3f} (42-44)")
    out.append(C)
    C = Check("BO4")
    for c in ("n1e-2", "n3e-2"):
        mean_rule(C, T(c, "bo", "FINS weak"), lambda m: m <= 1.2, f"{c}")
    out.append(C)
    C = Check("BO5")
    mean_rule(C, T("n1e-2", "bo", "FINS pointwise"), lambda m: m >= 1.5, "pointwise 1e-2")
    out.append(C)
    C = Check("BO6")
    for row in ("LieNLSD_F", "LieNLSD_F@eval"):
        mean_rule(C, T("clean", "bo", row), lambda m: m >= 0.5, row, FIT)
    out.append(C)
    return out


def ks(T):
    out, conds = [], ("clean", "n1e-2", "n3e-2", "n5e-2", "n1e-1")
    C = Check("KS1")
    for c in conds:
        seed_rule(C, T(c, "ks", "FINS weak"), lambda x: x <= 0.05, 9, c)
    out.append(C)
    C = Check("KS2")
    for c in conds:
        seed_rule(C, T(c, "ks", "FINS weak, own count", "finite", "k"), lambda x: round(x) == 3, 8, c)
    out.append(C)
    C = Check("KS3")
    for c in conds:
        mean_rule(C, T(c, "ks", "FINS pointwise"), (lambda m: m <= 0.05) if c == "clean" else (lambda m: m >= 1.0), c)
    out.append(C)
    C = Check("KS4")
    for c in conds:
        for row in TS:
            seed_rule(C, T(c, "ks", row), lambda x: x <= 0.05, 9, f"{c} {row[11:]}")
    out.append(C)
    C = Check("KS5")
    for c in conds[1:]:
        seed_rule(C, T(c, "ks", TS05_HARD), lambda x: x >= 0.5, 5, c)
    out.append(C)
    return out


def nl(B, K):
    out = []
    for name, T, pde, bf, bn in (("NL1", B, "bo", 0.3, 0.2), ("NL2", K, "ks", 0.3, 0.3)):
        C = Check(name)
        for row in LEARNED:
            mean_rule(C, T("clean", pde, row), lambda m: m <= bf, f"{row.split(',')[0]} finite", FIT)
            mean_rule(C, T("clean", pde, row, "non-trivial"), lambda m: m <= bn, f"{row.split(',')[0]} non-trivial", FIT)
        if name == "NL1":
            a, b = T("clean", pde, LEARNED[0]), T("clean", pde, LEARNED[1])
            if C.has(a, b, seeds=FIT):
                C.test(mean(a, FIT) <= mean(b, FIT), "weak <= pointwise")
        out.append(C)
    C = Check("NL3")
    for row, za in zip(LEARNED, ZERO_MEAN):
        a, m = K("clean", "ks", za, "non-trivial"), K("clean", "ks", row, "non-trivial")
        if C.has(a, m, seeds=FIT):
            C.test(mean(a, FIT) >= 0.5, f"{za.split(',')[0]} zero-mean {mean(a, FIT):.3f} (mixed {mean(m, FIT):.3f})")
            C.test(all(a[s] > m[s] for s in FIT), "above 'mixed' on " + f"{sum(a[s] > m[s] for s in FIT)}/3 seeds")
    out.append(C)
    C = Check("NL4")
    for T, pde, bound in ((B, "bo", 1.5), (K, "ks", 1.0)):
        for row in ("LieGAN", "LaLiGAN"):
            mean_rule(C, T("clean", pde, row), lambda m, b=bound: m >= b, f"{pde} {row} (>= {bound})", FIT)
    out.append(C)
    C = Check("NL5")
    for T, pde in ((B, "bo"), (K, "ks")):
        a, u = T("clean", pde, "Augerino"), T("clean", pde, "Augerino, untrained")
        if C.has(a, u, seeds=FIT):
            d = max(abs(a[s] - u[s]) for s in FIT)
            C.test(d <= 0.05, f"{pde} max |trained - untrained| {d:.3f}")
    out.append(C)
    C = Check("NL6")
    for T, pde, rivals in ((B, "bo", ("LieGAN", "LaLiGAN", "Augerino")),
                           (K, "ks", ("LieGAN", "LaLiGAN", "Augerino", "LIG"))):
        for row in LEARNED:
            f = T("clean", pde, row)
            for r in rivals:
                g = T("clean", pde, r)
                seeds = [s for s in FIT if s in g] if r == "LIG" else FIT
                if not seeds:
                    C.pending = True
                    continue
                if C.has(f, g, seeds=seeds):
                    C.test(mean(f, seeds) < mean(g, seeds),
                           f"{pde} {row.split(',')[0]} {mean(f, seeds):.3f} < {r} {mean(g, seeds):.3f}")
    out.append(C)
    C = Check("NL7")
    v = K("clean", "ks", "LIG")
    seeds = [s for s in FIT if s in v]
    if seeds:
        C.test(mean(v, seeds) >= 1.0, f"LIG {mean(v, seeds):.3f} on {len(seeds)} seed(s)")
    else:
        C.pending = True
    out.append(C)
    C = Check("NL8")
    for c in ("clean", "n1e-2", "n3e-2", "n5e-2", "n1e-1"):
        seed_rule(C, K(c, "ks", "LieNLSD_RHS@eval"), lambda x: x <= 0.05, 9, c)
    out.append(C)
    C = Check("NL9")
    for c, rule, txt in (("clean", lambda m: m <= 0.1, "<= 0.1"), ("n1e-2", lambda m: m >= 0.3, ">= 0.3")):
        for row in ("LieNLSD_F", "LieNLSD_F@eval"):
            mean_rule(C, K(c, "ks", row), rule, f"{c} {row} ({txt})", FIT)
    out.append(C)
    return out


def ex(W, B, U):
    """EX1-EX3 (exp55): the two-stage pipeline on Burgers and with the library to u_xxxx."""
    out = []
    o4 = [f"two-stage, H {h}, library to u_xxxx, {r}" for h in ("0.2", "0.05") for r in ("soft", "hard (own count)")]
    C = Check("EX1")
    rows = [f"two-stage, H {h}, {r}" for h in ("0.2", "0.05") for r in ("soft", "hard (own count)")] + o4
    for row in rows:
        seed_rule(C, U("clean", "burgers", row), lambda x: x <= 0.05, 9, f"Burgers {row[11:]}")
    out.append(C)
    C = Check("EX2")
    for pde in ("heat", "kdv"):
        for c in ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1"):
            for row in o4[:2]:
                seed_rule(C, W(c, pde, row), lambda x: x <= 0.05, 9, f"{pde} {c} {row[11:]}")
    out.append(C)
    C = Check("EX3")
    for c in ("clean", "n1e-2", "n3e-2"):
        for row in (o4[1], o4[3]):
            seed_rule(C, B(c, "bo", row), lambda x: abs(x - 2) < 1e-3, 10, f"{c} {row[11:]}")
    out.append(C)
    C = Check("EX4")
    mean_rule(C, W("clean", "heat", LEARNED[0]), lambda m: m <= 0.3, "heat learned weak finite (<= 0.3)", FIT)
    mean_rule(C, W("clean", "heat", LEARNED[0], "full"), lambda m: m >= 1.0, "class (>= 1.0)", FIT)
    out.append(C)
    return out


def lw(W, B, K, U, X):
    """LW1-LW5 (exp56, notebook 5): reference weak SINDy then Lie, exponent readout, learned operators under noise."""
    out = []
    eqs = ((W, "heat", HEAT_KDV, ("finite", "full")), (W, "kdv", HEAT_KDV, ("finite",)),
           (K, "ks", KS_CONDS, ("finite",)), (U, "burgers", ("clean",), ("finite",)))
    C = Check("LW1")
    for T, pde, conds, cvs in eqs:
        C.group([(f"{c} {row[24:]} {cv}", T(c, pde, row, cv)) for c in conds for row in WS_EVAL for cv in cvs],
                lambda x: x <= 0.05, 9, pde)
    out.append(C)
    C = Check("LW2")
    for T, pde, conds, cvs in eqs:
        conds = ("clean",) if pde == "burgers" else ("clean", "n1e-2")
        C.group([(f"{c} {row[22:]}", T(c, pde, row)) for c in conds for row in WS_TRAIN], lambda x: x <= 0.05, 3, pde,
                FIT)
        if pde == "heat":
            v = [T(c, pde, row, "full") for c in conds for row in WS_TRAIN]
            if all(all(q in x for q in FIT) for x in v):
                C.obs.append("heat class (not registered) worst " + f"{max(max(x[q] for q in FIT) for x in v):.3f}")
    out.append(C)
    C = Check("LW3")
    cells = [(f"{c} evaluation", X(c, "bo", WS_EVAL[1]), SEEDS) for c in ("clean", "n1e-2", "n3e-2")]
    cells += [(f"{c} training", X(c, "bo", WS_TRAIN[1]), FIT) for c in ("clean", "n1e-2")]
    for lab, v, seeds in cells:
        if C.has(v, seeds=seeds):
            ok = [v[q][0] >= 2.5 or v[q][1] < 0.9 for q in seeds]
            C.test(all(ok), f"{lab} {sum(ok)}/{len(seeds)} (a-hat {min(v[q][0] for q in seeds):.2f}-"
                            f"{max(v[q][0] for q in seeds):.2f}, containment {min(v[q][1] for q in seeds):.2f}-"
                            f"{max(v[q][1] for q in seeds):.2f})")
    out.append(C)
    C = Check("LW4")
    for c, ref in (("n1e-3", 1.68), ("n1e-2", 2.96)):
        mean_rule(C, W(c, "heat", LEARNED[0]), lambda m, b=ref / 5: m <= b, f"heat {c} (<= {ref / 5:.3f})", FIT)
    out.append(C)
    C = Check("LW5")
    for T, pde in ((B, "bo"), (K, "ks")):
        mean_rule(C, T("n1e-2", pde, LEARNED[0]), lambda m: m <= 0.3, f"{pde} weak finite", FIT)
        mean_rule(C, T("n1e-2", pde, LEARNED[0], "non-trivial"), lambda m: m <= 0.2, f"{pde} weak non-trivial", FIT)
        mean_rule(C, T("n1e-2", pde, LEARNED[1]), lambda m: m >= 1.0, f"{pde} pointwise finite", FIT)
    v = X("n1e-2", "bo", LEARNED[0])
    if C.has(v, seeds=FIT):
        n = sum(abs(v[q][0] - 2) <= 0.2 for q in FIT)
        C.test(n == 3, "bo weak a-hat " + " ".join(f"{v[q][0]:.3f}" for q in FIT))
    out.append(C)
    return out


def fbv(F, FX, O):
    """FB1-FB5 (exp57, notebook 6): fractional Burgers, the exponent read from the symmetry."""
    out, pdes = [], {"fb125": 1.25, "fb150": 1.5, "fb175": 1.75, "fb200": 2.0}
    near = (lambda al, tol: lambda v: abs(v[0] - al) <= tol)
    C = Check("FB1")
    for pde, al in pdes.items():
        seed_rule(C, FX("clean", pde, "FINS weak"), near(al, 0.1), 9, f"{pde} a-hat")
        seed_rule(C, F("clean", pde, "FINS weak"), lambda x: x <= 0.05, 9, f"{pde} finite")
    out.append(C)
    C = Check("FB2")
    for pde, al in pdes.items():
        seed_rule(C, FX("n1e-2", pde, "FINS weak"), near(al, FB2_TOL), 8, f"{pde} a-hat")
    out.append(C)
    C = Check("FB3")
    hard = "WSINDy, evaluation data, hard (own count)"
    seed_rule(C, FX("clean", "fb200", hard), near(2.0, 0.05), 9, "fb200 clean a-hat")
    for c in ("clean", "n1e-2"):
        for pde, al in list(pdes.items())[:3]:
            seed_rule(C, FX(c, pde, hard), lambda v, al=al: abs(v[0] - al) > 0.1 or v[1] < 0.9, 9, f"{pde} {c} absent")
    out.append(C)
    C = Check("FB4")
    for pde, al in pdes.items():
        v = O.get("clean", {}).get(pde, {})
        want = [] if al == 2 else [al]
        ok = {q: r["mult"] == want and r["balance"] is not None and abs(r["balance"] - al) < 1e-9
              for q, r in v.items()}
        seed_rule(C, ok, bool, 9, pde)
    out.append(C)
    C = Check("FB5")
    for pde, al in list(pdes.items())[:3]:
        seed_rule(C, FX("clean", pde, LEARNED[0]), near(al, 0.15), 3, f"{pde} learned a-hat", FIT)
        mean_rule(C, F("clean", pde, LEARNED[0]), lambda m: m <= 0.3, f"{pde} learned finite", FIT)
    out.append(C)
    return out


class Sym:
    """{seed: row} of exp58's summary: (condition, equation, row)."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, c, pde, row):
        return self.d.get(pde, {}).get(c, {}).get(row, {})


def slv(S):
    """SL1-SL6 (exp58): symmetry-informed weak SINDy."""
    out = []
    true = (lambda r: r["true_eq"])
    outside = [("bo", c) for c in ("clean", "n1e-2", "n3e-2")] + [(p, c) for p in ("fb125", "fb150", "fb175")
                                                                    for c in ("clean", "n1e-2")]
    C = Check("SL1")
    C.group([(f"{p} {c}", S(c, p, "sym")) for p, c in outside], true, 9, "sym true equation")
    out.append(C)
    C = Check("SL2")
    C.group([(f"{p} {c}", S(c, p, "prof-free")) for p, c in outside], true, 9, "prof-free true equation")
    worse = []
    for p, c in outside:
        a, b = S(c, p, "prof-free"), S(c, p, "sym")
        if C.has(a, b):
            ma = np.median([r["beta_err"] for r in a.values() if r["beta_err"] is not None] or [np.inf])
            mb = np.median([r["beta_err"] for r in b.values() if r["beta_err"] is not None] or [np.inf])
            if ma > mb:
                worse.append(f"{p} {c} {ma:.4f} > {mb:.4f}")
    C.test(not worse, "median |beta - alpha| of prof-free <= sym's in every cell" + (": not in " + ", ".join(worse)
                                                                                      if worse else ""))
    out.append(C)
    C = Check("SL3")
    C.group([(f"kdv {c}", S(c, "kdv", "sym")) for c in HEAT_KDV[:5]], true, 9, "sym true equation")
    v = S("n1e-1", "kdv", "sym")
    if C.has(v):
        want = {"43", "47", "50"}
        got = {q for q in SEEDS if v[q]["true_eq"]}
        C.test(got == want, f"1e-1 true on {sorted(got)} (predicted {sorted(want)})")
    out.append(C)
    C = Check("SL4")
    C.group([(f"heat {c}", S(c, "heat", "sym")) for c in HEAT_KDV], true, 9, "sym true equation")
    for c, q in (("n3e-2", "49"), ("n1e-1", "44")):
        v = S(c, "heat", "sym")
        if C.has(v, seeds=[q]):
            C.test(v[q]["true_eq"], f"{c} seed {q} {'true' if v[q]['true_eq'] else 'wrong'}")
    out.append(C)
    C = Check("SL5")
    C.group([(f"ks {c}", S(c, "ks", "sym")) for c in KS_CONDS], true, 9, "sym true equation")
    C.group([(f"ks {c}", S(c, "ks", "prof")) for c in KS_CONDS], lambda r: not r["true_eq"], 10, "prof wrong")
    C.group([(f"ks {c}", S(c, "ks", "prof-free")) for c in KS_CONDS], true, 9, "prof-free true equation")
    out.append(C)
    C = Check("SL6")
    C.group([(f"fb200 {c}", S(c, "fb200", "sym")) for c in ("clean", "n1e-2")], true, 9, "sym true equation")
    out.append(C)
    return out


class Aniso:
    """{seed: row} of exp60's summary: (exponent, row); each row carries its exponent as "a"."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, a, row):
        return {s: dict(r, a=float(a)) for s, r in self.d.get(a, {}).get(row, {}).items()}


def within(x, y, tol):
    """|x - y| <= tol for a readout x (None fails); 1e-9 absorbs the decimal boundary (0.61 - 0.6 > 0.01 in floats)."""
    return x is not None and abs(x - y) <= tol + 1e-9


def anv(A):
    """AN1-AN5 (exp60): anisotropic nonlocal dissipation in 2D, per exponent a."""
    out = []
    cells = (lambda row: [(f"a {a}", A(a, row)) for a in AN_EXPONENTS])
    no_rot = (lambda r: r["eps_hat"] is None)
    scans = (lambda te, ta: (lambda r: within(r["scan"]["eps"], AN_EPS, te) and within(r["scan"]["a"], r["a"], ta)))
    C = Check("AN1")
    C.group(cells("FINS exact"), scans(0.01, 0.02), 9, "FINS exact scans, eps-hat within 0.01 and a-hat within 0.02")
    out.append(C)
    C = Check("AN2")
    C.group(cells("iso-free"), lambda r: r["a_hat"] is None or not within(r["prm"][0], r["a"], 0.1), 8,
            "iso-free |beta - a| > 0.1 or no balance")
    C.group(cells("iso-free"), no_rot, 10, "iso-free no stretched rotation")
    out.append(C)
    C = Check("AN3")
    C.group(cells("harm-free"), lambda r: within(r["a_hat"], r["a"], 0.05), 9, "harm-free a-hat within 0.05")
    C.group(cells("harm-free"), no_rot, 10, "harm-free no stretched rotation")
    out.append(C)
    C = Check("AN4")
    C.group(cells("ell-free"), lambda r: r["true_eq"], 9, "ell-free true equation")
    out.append(C)
    C = Check("AN5")
    C.group(cells("FINS FNO2d"), scans(0.05, 0.1), 2, "FINS FNO2d scans, eps-hat within 0.05 and a-hat within 0.1",
            seeds=FIT)
    out.append(C)
    return out


PG_REGIMES = ("f", "beta", "rd", "drag")
PG_CAND = ("d_x", "d_y", "d_t", "shift", "rotation", "D_x", "D_t", "D_beta")
PG_KEPT = {"f": PG_CAND, "beta": ("d_x", "d_y", "d_t", "shift", "D_beta"),
           "rd": ("d_x", "d_y", "d_t", "shift", "rotation", "D_t"),
           "drag": ("d_x", "d_y", "d_t", "shift", "rotation", "D_x")}


class Pyqg:
    """{seed: row} of exp61's summary by label ("exact q f", "fno q beta", ...); rows carry their regime."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, label):
        return {s: dict(r, regime=label.split()[2]) for s, r in self.d.get(label, {}).items()}


def pg_kept_max(r):
    return max(r["quotients"][c] for c in PG_KEPT[r["regime"]])


def pg_broken_ok(r):
    br = [c for c in PG_CAND if c not in PG_KEPT[r["regime"]]]
    return not br or min(r["quotients"][c] for c in br) >= 100 * pg_kept_max(r)


def pg_span_missing(vecs, targets):
    """Missing dimensions of targets (unit-normalised) in the span of vecs, Euclidean in the 3 scaling coordinates."""
    Q, _ = np.linalg.qr(np.array(vecs, dtype=float).T)
    T = np.array(targets, dtype=float)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    return float(len(T) - ((Q.T @ T.T) ** 2).sum())


def pg_scaling_ok(r, tol=0.02):
    lam, vec, reg = r["planes"]["scal_lam"], r["planes"]["scal_vec"], r["regime"]
    s = 2 if reg == "f" else 1
    if not (lam[s - 1] <= 1e-4 and lam[s] >= 10 * lam[s - 1]):
        return False
    w = vec[0]
    if reg == "f":
        return pg_span_missing(vec[:2], [(1, 0, 0), (0, 1, -1)]) <= 0.05
    if reg == "beta":
        return within(w[1] / w[0], -1, tol) and within(w[2] / w[0], 1, tol)
    if reg == "rd":
        return within(w[0] / w[1], 0, tol) and within(w[2] / w[1], -1, tol)
    return within(w[1] / w[0], 0, tol) and within(w[2] / w[0], 0, tol)


def pgv(P):
    """PG1-PG8 (exp61): quasi-geostrophic flow from pyqg."""
    out = []
    q = [(reg, P(f"exact q {reg}")) for reg in PG_REGIMES]
    C = Check("PG1")
    C.group(q, lambda r: r["span_k"]["missing"] <= 0.05 + 1e-9, 9, "q missing <= 0.05")
    out.append(C)
    C = Check("PG2")
    C.group(q, lambda r: r["own"]["floor6"] == len(r["names"]), 9, "own count (floor 1e-6) = k")
    out.append(C)
    C = Check("PG3")
    C.group(q, lambda r: pg_kept_max(r) <= 1e-4, 9, "kept candidates <= 1e-4")
    C.group(q[1:], pg_broken_ok, 9, "broken >= 100 x the largest kept")
    out.append(C)
    C = Check("PG4")
    C.group(q, pg_scaling_ok, 9, "scaling space: count, gap and directions")
    out.append(C)
    C = Check("PG5")
    C.group(q[1:2], lambda r: r["planes"]["rot_lam"][0] >= 1e-3, 10, "rotation plane >= 1e-3 at beta")
    C.group([q[0], q[2], q[3]], lambda r: r["planes"]["rot_lam"][0] <= 1e-4 and r["planes"]["rot_e"] is not None
            and within(r["planes"]["rot_e"], 1.0, 0.01), 9, "rotation plane <= 1e-4, stretch 1 +- 0.01 at f, rd, drag")
    out.append(C)
    C = Check("PG6")
    C.group([("f n64", P("exact q f n64"))], lambda r: r["quotients"]["D_x"] >= 1e-3, 9, "D_x >= 1e-3 at 64^2")
    out.append(C)
    C = Check("PG7")
    C.group([("psi f", P("exact psi f"))], lambda r: r["quotients"]["D_x"] >= 1e-2 and r["quotients"]["D_t"] <= 1e-4
            and r["quotients"]["rotation"] <= 1e-3, 9, "psi f: D_x >= 1e-2, D_t <= 1e-4, rotation <= 1e-3")
    C.group([("psi rd", P("exact psi rd"))], lambda r: r["span_k"]["missing"] <= 0.05 + 1e-9
            and r["own"]["floor6"] == 6 and r["quotients"]["D_x"] >= 100 * max(
                r["quotients"][c] for c in ("d_x", "d_y", "d_t", "shift", "rotation", "D_t")), 9,
            "psi rd: missing, own count 6, D_x >= 100 x kept")
    out.append(C)
    C = Check("PG8")
    fb, ff = P("fno q beta"), P("fno q f")
    if C.has(fb, ff, seeds=FIT):
        n = sum(fb[s]["planes"]["rot_lam"][0] >= 10 * ff[s]["planes"]["rot_lam"][0] for s in FIT)
        C.test(n >= 2, f"rotation plane beta >= 10 x f: {n}/3")
    C.group([("fno beta", fb)], lambda r: within(r["planes"]["scal_vec"][0][1] / r["planes"]["scal_vec"][0][0], -1, 0.1)
            and within(r["planes"]["scal_vec"][0][2] / r["planes"]["scal_vec"][0][0], 1, 0.1), 2,
            "fno beta: scaling direction within 0.1", seeds=FIT)
    C.group([("fno f", ff)], lambda r: pg_span_missing(r["planes"]["scal_vec"][:2], [(1, 0, 0), (0, 1, -1)]) <= 0.2, 2,
            "fno f: two deepest scaling directions span D_x, D_t", seeds=FIT)
    out.append(C)
    return out


class PyqgWS:
    """{seed: row} of exp61's weak SINDy summary: (regime, row)."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, regime, row):
        return self.d.get(regime, {}).get(row, {})


def wsv(W):
    """WS1-WS5 (exp61, weak SINDy arm)."""
    out = []
    cells = (lambda row, regs=PG_REGIMES: [(reg, W(reg, row)) for reg in regs])
    C = Check("WS1")
    C.group(cells("ref"), lambda r: not r["true_eq"], 10, "ref not the true equation")
    C.group(cells("ref"), lambda r: not r["right_algebra"], 9, "ref algebra differs from the regime's")
    out.append(C)
    C = Check("WS2")
    C.group(cells("qg-free", ("f", "beta", "drag")), lambda r: r["true_eq"], 9, "qg-free true equation")
    C.group(cells("qg-free", ("rd",)), lambda r: not r["true_eq"], 9, "qg-free not the true equation at rd")
    out.append(C)
    C = Check("WS3")
    C.group(cells("qg-rd"), lambda r: r["true_eq"], 9, "qg-rd true equation")
    out.append(C)
    C = Check("WS4")
    C.group(cells("nested"), lambda r: r["true_eq"], 9, "nested true equation")
    out.append(C)
    C = Check("WS5")
    C.group(cells("nested", ("f", "beta", "rd")), lambda r: r["true_eq"] and r["right_algebra"], 9,
            "nested true equation with the regime's algebra")
    C.group(cells("nested", ("drag",)), lambda r: r["true_eq"] and not r["algebra"]["shift"], 9,
            "drag: nested true equation breaking the shift")
    out.append(C)
    return out


DT_ROWS = (("drag", 0.1), ("drag", 0.3), ("drag", 0.6), ("f", 0.0))


class DragTau:
    """{seed: row} of exp62's summary by (regime, rek)."""

    def __init__(self, path):
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __call__(self, regime, rek):
        return {s: dict(r, rek=rek) for s, r in self.d.get(f"{regime} rek {rek:g}", {}).items()}


def vr_exact(r):
    return next(v for k, v in r["quotients"].items() if k.endswith("exact"))


def dtv(G):
    """DT1-DT5 (exp62): the drag's time scaling in a class with tau(t) to degree 6."""
    rows = [(f"{reg} {rek:g}", G(reg, rek)) for reg, rek in DT_ROWS]
    out = []
    C = Check("DT1")
    C.group(rows, lambda r: r["own"] == 7, 9, "own count (floor 1e-6) = 7")
    out.append(C)
    C = Check("DT2")
    C.group(rows, lambda r: r["span_k"]["missing"] <= 0.05 + 1e-9, 9, "missing <= 0.05 (span of the 7 lowest)")
    out.append(C)
    C = Check("DT3")
    C.group(rows, lambda r: within(r["r_hat"]["phi"], r["rek"], 0.01), 9, "r-hat (phi) within 0.01 of rek")
    out.append(C)
    C = Check("DT4")
    C.group(rows[:3], lambda r: vr_exact(r) <= 1e-6 and r["quotients"]["D_t"] >= 1e-2, 9,
            "drag: V_r exact <= 1e-6, D_t >= 1e-2")
    C.group(rows[3:], lambda r: vr_exact(r) >= 1e-2, 9, "f: V_0.3 exact >= 1e-2")
    out.append(C)
    C = Check("DT5")
    C.group(rows, lambda r: r["time"]["lam"][1] <= 1e-6 and r["time"]["lam"][2] >= 1e-2, 9,
            "time pencil: two eigenvalues <= 1e-6, the third >= 1e-2")
    out.append(C)
    return out


def arv(path):
    """AR1-AR2 (exp60_readouts.py): the sl(2) pencil of exp60's exact runs."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    rows = [(a, {s: r["sl2"] for s, r in d.get(a, {}).items() if "sl2" in r}) for a in ("1.25", "1.5", "1.75")]
    C1, C2 = Check("AR1"), Check("AR2")
    C1.group(rows, lambda r: r["det_sign"] > 0 and within(r["eps"], 0.6, 0.005) and within(r["angle"], 0.0, 0.5), 9,
             "sl(2): elliptic, eps-hat within 0.005, angle within 0.5 deg")
    C2.group(rows, lambda r: r["lam"][1] >= 1000 * r["lam"][0], 9, "sl(2): lambda2 / lambda1 >= 1000")
    return [C1, C2]


EA_GENS = {"f": ("d_x", "d_y", "rotation", "D_x", "D_t"), "beta": ("d_x", "d_y", "D_beta")}


def eav(path):
    """EA1-EA3 (exp65): PG8's emulators audited on their own rollouts and on transformed inputs."""
    d = json.load(open(path)) if os.path.exists(path) else {"fins": {}, "consequence": {}}
    fi, co = d.get("fins", {}), d.get("consequence", {})
    C1, C2, C3 = Check("EA1"), Check("EA2"), Check("EA3")
    ff, fb = fi.get("f", {}), fi.get("beta", {})
    if C1.has(ff, fb, seeds=FIT):
        n = sum(fb[s]["planes"]["rot_lam"][0] >= 10 * ff[s]["planes"]["rot_lam"][0] for s in FIT)
        C1.test(n >= 2, f"self-consistent rotation plane beta >= 10 x f: {n}/3")
    C2.group([("beta", fb)], lambda r: within(r["planes"]["scal_vec"][0][1] / r["planes"]["scal_vec"][0][0], -1, 0.1)
             and within(r["planes"]["scal_vec"][0][2] / r["planes"]["scal_vec"][0][0], 1, 0.1), 2,
             "self-consistent beta scaling direction within 0.1", seeds=FIT)
    cf, cb = co.get("f", {}), co.get("beta", {})
    if C3.has(ff, fb, cf, cb, seeds=FIT):
        x = [fi[r][s]["quotients"][g] for r in EA_GENS for s in FIT for g in EA_GENS[r]]
        y = [co[r][s]["ratio"][g] for r in EA_GENS for s in FIT for g in EA_GENS[r]]
        rho = float(spearmanr(x, y)[0])
        C3.test(rho >= 0.6, f"Spearman (quotient, error ratio) over {len(x)} pairs {rho:.3f}")
    return [C1, C2, C3]


def lsv(path):
    """LS1-LS2 (exp66): LieNLSD as published on exp60."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    rows = [(a, {s: dict(r["fitted"], a=float(a)) for s, r in d.get(a, {}).items() if "fitted" in r})
            for a in ("1.25", "1.5", "1.75")]
    C1, C2 = Check("LS1"), Check("LS2")
    C1.group(rows, lambda r: abs(r["scan"]["a"] - float(r["a"])) > 0.1, 8, "a-hat more than 0.1 off")
    C2.group(rows, lambda r: r["missing_5"] >= 0.5, 8, "span of 5 misses >= 0.5")
    return [C1, C2]


def epv(path):
    """EP0-EP5 (exp68): emulators of pyqg with the improved recipe; rotation augmentation and grid channels."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    cell = (lambda reg, arm: d.get(reg, {}).get(arm, {}))
    err = (lambda r: r["error"]["held-out"]["one"])
    rlam = (lambda r: r["fins"]["planes"]["rot_lam"][0])
    com = (lambda r, k: r["commute"]["emulator"][k]["rms"])
    fp, bp, fr, fg = cell("f", "plain"), cell("beta", "plain"), cell("f", "rot"), cell("f", "grid")
    out = [Check(f"EP{i}") for i in range(6)]
    out[0].group([("f plain", fp), ("beta plain", bp)], lambda r: err(r) <= 0.03, 3, "held-out one-step error <= 0.03",
                 seeds=FIT)
    if out[1].has(fp, bp, seeds=FIT):
        n = sum(rlam(bp[s]) >= 10 * rlam(fp[s]) for s in FIT)
        out[1].test(n >= 2, f"rotation plane beta >= 10 x f: {n}/3")
    vec = (lambda r: r["fins"]["planes"]["scal_vec"][0])
    out[2].group([("beta plain", bp)], lambda r: within(vec(r)[1] / vec(r)[0], -1, 0.1)
                 and within(vec(r)[2] / vec(r)[0], 1, 0.1), 2, "beta scaling direction within 0.1", seeds=FIT)
    out[3].group([("f plain", fp), ("f rot", fr), ("f grid", fg)],
                 lambda r: 0.35 / 3 <= rlam(r) / err(r) ** 2 <= 0.35 * 3, 3,
                 "rotation plane within a factor 3 of 0.35 e^2", seeds=FIT)
    if out[4].has(fp, fr, seeds=FIT):
        n1 = sum(com(fr[s], "rotation 30") <= com(fp[s], "rotation 30") / 1.5 and rlam(fr[s]) <= rlam(fp[s]) / 2
                 for s in FIT)
        n2 = sum(abs(np.log((rlam(fp[s]) / rlam(fr[s])) / (err(fp[s]) / err(fr[s])) ** 2)) <= np.log(1.5) for s in FIT)
        out[4].test(n1 == 3, f"rot: 30-degree commutator 1.5 x and rotation plane 2 x lower: {n1}/3")
        out[4].test(n2 >= 2, f"rot: the rotation-plane ratio within 1.5 x of the squared error ratio: {n2}/3")
    if out[5].has(fp, fg, seeds=FIT):
        n1 = sum(min(com(fg[s], "d_x 32"), com(fg[s], "d_y 32")) >= 1e-2 and max(com(fp[s], "d_x 32"),
                                                                              com(fp[s], "d_y 32")) <= 1e-6 for s in FIT)
        q = (lambda r, g: r["fins"]["quotients"][g])
        n2 = sum(all(1 / 3 <= q(fg[s], g) / q(fp[s], g) <= 3 for g in ("d_x", "d_y")) for s in FIT)
        n3 = sum(abs(np.log(err(fg[s]) / err(fp[s]))) <= np.log(1.5) for s in FIT)
        out[5].test(n1 == 3, f"grid: 32-cell commutators >= 1e-2, plain's <= 1e-6: {n1}/3")
        out[5].test(n2 >= 2, f"grid: FINS translations within 3 x of plain's: {n2}/3")
        out[5].test(n3 >= 2, f"grid: held-out errors within 1.5 x: {n3}/3")
    return out


TR_SEEDS = [str(s) for s in range(42, 52)]
TR_OBS = ("d_x", "d_y", "d_t", "rotation", "d_c", "c_dc")
TR_HEAT = ("boost_x", "boost_y", "sup_x", "sup_y", "sup_xy", "sup_x2-y2", "sup_r2")


def trv(path):
    """TR0-TR8 (exp69): a passive tracer in a hidden flow; the simulator oracle, weak SINDy on c, window emulators, the
    best hidden lift."""
    d = json.load(open(path)).get("runs", {}) if os.path.exists(path) else {}
    orc, reg, obs = d.get("oracle", {}), d.get("regress", {}), d.get("observability", {})
    # the self arm in float64 (notebook 11b, seeds 42-44); notebook 11's float32 runs (43, 44) agree to two digits
    # above the round-off floor and give the same verdicts
    sf, df = d.get("self64 fno w4", {}), d.get("data fno w4", {})
    q = (lambda r, k: r["quotients"][k])
    mobs = (lambda r: max(q(r, k) for k in TR_OBS))
    out = [Check(f"TR{i}") for i in range(9)]
    out[0].group([("self64 fno w4", sf)], lambda r: r["error_one"] <= 0.02, 3, "held-out one-step error <= 0.02",
                 seeds=FIT)
    out[1].group([("oracle", orc)], lambda r: r["own"] == 7 and r["span7"]["missing7"] <= 0.05, 9,
                 "own count 7, the span of 7 misses <= 0.05", seeds=TR_SEEDS)
    out[2].group([("oracle", orc)], lambda r: abs(r["S_lift"] + 2) <= 0.02 and r["S_unlifted"] >= 100 * q(r, "S"), 9,
                 "S's lift within 0.02 of -2, unlifted >= 100 x lifted", seeds=TR_SEEDS)
    out[3].group([("oracle", orc)], lambda r: min(q(r, k) for k in TR_HEAT) >= 100 * max(mobs(r), q(r, "S")), 9,
                 "heat-false >= 100 x the observed algebra and S", seeds=TR_SEEDS)
    out[4].group([("regress", reg)], lambda r: r["algebra"]["in_H"] == 7 and r["algebra"]["outside_obs7"] >= 7, 9,
                 "the identified algebra holds H's seven, >= 7 outside the observed algebra with S", seeds=TR_SEEDS)
    out[5].group([("regress", reg)], lambda r: not r["algebra"]["contains"]["rotation"], 9, "rotation lost",
                 seeds=TR_SEEDS)
    out[6].group([("self", sf), ("data", df)], lambda r: r["H_lam"][0] <= 3 * mobs(r), 2,
                 "H's lowest <= 3 x the observed algebra's largest", seeds=FIT)
    if out[7].has(sf, orc, seeds=FIT):
        n = sum(all(q(sf[s], k) <= 0.3 * q(orc[s], k) for k in ("sup_x", "sup_y", "boost_x", "boost_y")) for s in FIT)
        out[7].test(n >= 2, f"emulator's sup_x, sup_y, boost_x, boost_y <= 0.3 x the oracle's: {n}/3")
    best = (lambda r, k: r["quotients"][k]["best"])
    out[8].group([("observability", obs)], lambda r: min(best(r, k) for k in TR_HEAT) >= 1e-5
                 and max(best(r, k) for k in TR_OBS + ("S",)) <= 1e-9, 3,
                 "best hidden lift: heat-false >= 1e-5, observed algebra and S <= 1e-9", seeds=FIT)
    return out


def hlv(path):
    """UF1-UF2, HL0-HL5 (exp70, step 1): the uniform flow in the best hidden lift; the emulator of the full state."""
    d = json.load(open(path)).get("runs", {}) if os.path.exists(path) else {}
    uf, tr8 = d.get("observe", {}), d.get("observability", {})
    sd, ss = d.get("state data", {}), d.get("state self", {})
    orc, win = d.get("oracle", {}), d.get("window data", {})
    q = (lambda r, k: r["quotients"][k])
    top = (lambda r: max(max(q(r, k) for k in TR_OBS), q(r, "S")))
    boosts = (lambda r: min(q(r, "boost_x"), q(r, "boost_y")))
    out = [Check(n) for n in ("UF1", "UF2", "HL0", "HL1", "HL2", "HL3", "HL4", "HL5")]
    out[0].group([("observe", uf)], lambda r: max(q(r, k)["best_u"] for k in ("gal_x", "gal_y")) <= 1e-12 and all(
        abs(f[0] - 1) <= 1e-3 for f in q(r, "gal_x")["flow"]) and all(abs(f[1] - 1) <= 1e-3 for f in q(r, "gal_y")["flow"]),
                 3, "gal_x, gal_y <= 1e-12 with the flow, their coefficients within 1e-3 of 1 on every pair", seeds=FIT)
    if out[1].has(uf, tr8, seeds=FIT):
        n = sum(all(q(uf[s], k)["best_u"] >= 0.5 * q(tr8[s], k)["best"] for k in TR_HEAT) for s in FIT)
        lo = min(q(uf[s], k)["best_u"] / q(tr8[s], k)["best"] for s in FIT for k in TR_HEAT)
        out[1].test(n == 3, f"every heat-false generator with the flow >= 0.5 x TR8's: {n}/3 (smallest ratio {lo:.2f})")
    out[2].group([("state data", sd)], lambda r: r["error_one"][1] <= 0.04 and r["error_one"][0] <= 0.05, 3,
                 "one-step error c <= 0.04, q <= 0.05", seeds=FIT)
    out[3].group([("state data", sd)], lambda r: boosts(r) >= 10 * top(r), 2,
                 "both boosts >= 10 x the observed algebra and S", seeds=FIT)
    out[4].group([("state data", sd)], lambda r: abs(r["S_lift"] + 2) <= 0.1, 2, "S's hidden lift within 0.1 of -2",
                 seeds=FIT)
    if out[5].has(sd, orc, win, seeds=FIT):
        n1 = sum(all(1 / 3 <= q(sd[s], k) / q(orc[s], k) <= 3 for k in ("boost_x", "boost_y")) for s in FIT)
        n2 = sum(all(q(sd[s], k) >= 10 * q(win[s], k) for k in ("boost_x", "boost_y")) for s in FIT)
        out[5].test(n1 >= 2 and n2 == 3, f"boosts within 3 x of the oracle's {n1}/3; >= 10 x the window emulator's {n2}/3")
    out[6].group([("state data", sd)], lambda r: min(q(r, k) for k in TR_HEAT) < 10 * top(r), 2,
                 "the smallest heat-false < 10 x the observed algebra and S", seeds=FIT)
    out[7].group([("state self", ss)], lambda r: r["drift"] <= 0.05 and boosts(r) >= 10 * top(r), 2,
                 "own rollouts within 0.05 of the data and both boosts >= 10 x", seeds=FIT)
    return out


def ltv(path):
    """LT0-LT5 (exp70, step 2): a model of c alone with a learned hidden field; the plane readout (each candidate with
    its best z d_z weight) for the data and self arms."""
    d = json.load(open(path)).get("runs", {}) if os.path.exists(path) else {}
    ld, ls, ll = d.get("latent data", {}), d.get("latent self", {}), d.get("latent lift", {})
    win = d.get("window data", {})
    pl = (lambda r, k: r["plane"][k]["lam"])
    top = (lambda r: max(pl(r, k) for k in TR_OBS + ("S",)))
    boosts = (lambda r: min(pl(r, "boost_x"), pl(r, "boost_y")))
    best = (lambda r, k: r["quotients"][k]["best"])
    out = [Check(f"LT{i}") for i in range(6)]
    out[0].group([("latent data", ld)], lambda r: r["error_one"] <= 0.02, 3, "held-out one-step error of c <= 0.02",
                 seeds=FIT)
    out[1].group([("latent data", ld)], lambda r: boosts(r) >= 10 * top(r), 2,
                 "plane: both boosts >= 10 x the observed algebra and S", seeds=FIT)
    if out[2].has(ld, win, seeds=FIT):
        n = sum(all(pl(ld[s], k) >= 10 * win[s]["quotients"][k] for k in ("boost_x", "boost_y")) for s in FIT)
        out[2].test(n == 3, f"plane: both boosts >= 10 x the window emulator's: {n}/3")
    out[3].group([("latent data", ld)], lambda r: min(pl(r, k) for k in TR_HEAT) < 10 * top(r), 2,
                 "plane: the smallest heat-false < 10 x the observed algebra and S", seeds=FIT)
    out[4].group([("latent self", ls)], lambda r: r["drift"] <= 0.1 and boosts(r) >= 10 * top(r), 2,
                 "own rollouts within 0.1 of the data and plane boosts >= 10 x", seeds=FIT)
    out[5].group([("latent lift", ll)], lambda r: max(best(r, k) for k in TR_HEAT) <= max(
        best(r, k) for k in TR_OBS + ("S",)), 2, "every heat-false best lift <= the observed algebra's and S's largest",
                 seeds=FIT)
    return out


MF = ["42", "43", "44", "45", "46"]
UV_SCAL = {"f": None, "beta": ((0, (1, 2, 3)), (-1.0, 2.0, 2.0), 0.15), "rd": ((1, (0, 2, 3)), (0.0, -1.0, -1.0), 0.02),
           "drag": ((0, (1, 2, 3)), (0.0, 1.0, 1.0), 0.02)}


def uv_scaling_ok(r, regime):
    lam, vecs = r["scaling"]["lam"], r["scaling"]["vecs"]
    s = 2 if regime == "f" else 1
    if lam[s] < 10 * lam[s - 1]:
        return False
    if regime == "f":
        return pg_span_missing(vecs[:2], [(1, 0, 1, 1), (0, 1, -1, -1)]) <= 0.05
    (i, js), want, tol = UV_SCAL[regime]
    w = vecs[0]
    return all(within(w[j] / w[i], x, tol) for j, x in zip(js, want))


def uv_rotation_ok(r):
    q = r["quotients"]
    rot = r["rotation"]
    return (rot["lam"][0] <= 1e-4 and within(rot["spin"], 1.0, 0.01) and rot["stretch"] is not None
            and within(rot["stretch"], 1.0, 0.01)
            and min(q["scalar rotation"], q["spin"], q["opposite spin"]) >= 100 * rot["lam"][0])


def uv_boosts_kept(r):
    b = r["boost"]
    return (b["lam"][1] <= 1e-6 and all(d["x"] is not None and d["y"] is not None and within(d["x"], 1.0, 0.01)
                                        and within(d["y"], 1.0, 0.01) for d in b["deep"]))


def uvv(path):
    """UV1-UV5 (exp63): FINS for systems on pyqg's velocity."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    P = {reg: d.get(f"pyqg {reg}", {}) for reg in ("f", "beta", "rd", "drag")}
    M = {reg: d.get(f"meanflow {reg}", {}) for reg in ("f", "beta", "rd")}
    out = []
    C = Check("UV1")
    C.group([(f"pyqg {r}", P[r]) for r in ("f", "rd", "drag")], lambda r: r["span_k"]["missing"] <= 0.05 + 1e-9, 9,
            "missing <= 0.05 (span of k)")
    C.group([(f"meanflow {r}", M[r]) for r in ("f", "beta")], lambda r: r["span_k"]["missing"] <= 0.05 + 1e-9, 4,
            "meanflow: missing <= 0.05", seeds=MF)
    out.append(C)
    C = Check("UV2")
    C.group([(f"pyqg {r}", P[r]) for r in ("f", "rd", "drag")], uv_rotation_ok, 9,
            "vector rotation <= 1e-4, spin and stretch 1 +- 0.01, scalar / spin / opposite >= 100 x")
    C.group([("pyqg beta", P["beta"])], lambda r: r["rotation"]["lam"][0] >= 1e-2, 10, "beta: rotation space >= 1e-2")
    out.append(C)
    C = Check("UV3")
    C.group([(f"pyqg {r}", dict((s, dict(v, regime=r)) for s, v in P[r].items())) for r in ("f", "rd", "drag")],
            lambda r: uv_scaling_ok(r, r["regime"]), 9, "scaling space: count, gap, directions")
    C.group([("meanflow beta", M["beta"])], lambda r: uv_scaling_ok(r, "beta"), 4,
            "meanflow beta: direction within 0.15 of (1, -1, 2, 2)", seeds=MF)
    out.append(C)
    C = Check("UV4")
    C.group([(f"pyqg {r}", P[r]) for r in P], lambda r: r["boost"]["lam"][0] >= 100 * max(
        r["quotients"][n] for n in r["names"]), 9, "pyqg: boosts >= 100 x the algebra's largest")
    out.append(C)
    C = Check("UV5")
    C.group([(f"meanflow {r}", M[r]) for r in ("f", "beta")], uv_boosts_kept, 4,
            "meanflow f, beta: two boosts <= 1e-6, speeds 1 +- 0.01", seeds=MF)
    C.group([("meanflow rd", M["rd"])], lambda r: r["boost"]["lam"][0] >= 30 * max(
        r["quotients"][n] for n in r["names"]), 4, "meanflow rd: boosts >= 30 x the algebra's largest", seeds=MF)
    out.append(C)
    return out


def qlv(path):
    """QL1-QL4 (exp64): two-stage rows without quasi-geostrophic knowledge on pyqg's data."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    regs = ("f", "beta", "rd", "drag")
    row = (lambda name: [(f"{r} {name}", d.get(r, {}).get(name, {})) for r in regs])
    C1, C2, C3, C4 = Check("QL1"), Check("QL2"), Check("QL3"), Check("QL4")
    C1.group(row("obs"), lambda r: r["true_eq"] and r["right_algebra"], 9, "obs: true equation and algebra")
    C2.group(row("riesz"), lambda r: r["true_eq"] and r["right_algebra"], 9, "riesz: true equation and algebra")
    C3.group(row("nested"), lambda r: r["true_eq"], 9, "nested: true equation")
    C4.group([("rd riesz-free", d.get("rd", {}).get("riesz-free", {}))], lambda r: not r["true_eq"], 9,
             "rd: riesz-free not the true equation")
    return [C1, C2, C3, C4]


def ctv(path):
    """CT1-CT2 (exp67): exp60's controls through the exact solver."""
    d = json.load(open(path)) if os.path.exists(path) else {}
    C1, C2 = Check("CT1"), Check("CT2")
    C1.group([("mismatch", d.get("mismatch", {}))], lambda r: r["scan"]["q_eps"] >= 30 * max(
        r["quotients"][n] for n in ("d_x", "d_y", "d_t", "scaling")) and within(r["scan"]["a"], 1.5, 0.02)
        and within(r["scan"]["c"], 0.5, 0.02), 9, "mismatch: rotation scan >= 30 x kept, scaling (1.5, 0.5)")
    C2.group([("cubic", d.get("cubic", {}))], lambda r: r["quotients"]["scaling"] >= 100 * max(
        r["quotients"][n] for n in ("d_x", "d_y", "d_t", "stretched rotation")) and abs(r["scan"]["a"] - 1.5) > 0.05
        and within(r["scan"]["eps"], 0.6, 0.002), 9, "cubic: (1.5, 0.5) >= 100 x kept, a-hat off by > 0.05, eps 0.6")
    return [C1, C2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    B, K = (Table(os.path.join(args.results, d, "summary.json")) for d in ("bo", "ks"))
    W, U = (Table(os.path.join(args.results, d, "summary.json")) for d in ("noise_weak", "burgers"))
    X = Exponent(os.path.join(args.results, "exponent", "summary.json"))
    F, FX = Table(os.path.join(args.results, "fburgers", "summary.json")), \
        Exponent(os.path.join(args.results, "fburgers", "exponent.json"))
    p = os.path.join(args.results, "fburgers", "oracle.json")
    O = json.load(open(p)) if os.path.exists(p) else {}
    S = Sym(os.path.join(args.results, "symlib", "summary.json"))
    A = Aniso(os.path.join(args.results, "aniso2d", "summary.json"))
    P = Pyqg(os.path.join(args.results, "pyqg", "summary.json"))
    PW = PyqgWS(os.path.join(args.results, "pyqg", "wsindy_summary.json"))
    G = DragTau(os.path.join(args.results, "drag_tau", "summary.json"))
    checks = (wn(W) + bo(B) + ks(K) + nl(B, K) + ex(W, B, U) + lw(W, B, K, U, X) + fbv(F, FX, O) + slv(S) + anv(A)
              + arv(os.path.join(args.results, "aniso2d", "readouts_summary.json")) + pgv(P) + wsv(PW) + dtv(G)
              + uvv(os.path.join(args.results, "velocity", "summary.json"))
              + qlv(os.path.join(args.results, "pyqg", "libraries_summary.json"))
              + ctv(os.path.join(args.results, "aniso2d", "controls_summary.json"))
              + eav(os.path.join(args.results, "pyqg", "audit_summary.json"))
              + lsv(os.path.join(args.results, "aniso2d", "lienlsd_summary.json"))
              + epv(os.path.join(args.results, "pyqg", "pairs_summary.json"))
              + trv(os.path.join(args.results, "tracer", "summary.json"))
              + hlv(os.path.join(args.results, "tracer", "exp70_summary.json"))
              + ltv(os.path.join(args.results, "tracer", "exp70_summary.json")))
    print("\n".join(C.line() for C in checks))


if __name__ == "__main__":
    main()
