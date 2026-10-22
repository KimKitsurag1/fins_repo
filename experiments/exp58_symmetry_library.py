"""Experiment 58 -- symmetry-informed weak SINDy: the library built from the algebra FINS measured without the equation,
the coefficients from weak SINDy (README_EXPERIMENTS.md, pre-registration of 08.10).

Lie's inverse problem for the measured algebra, read from FINS's span by containment (at least GATE; FINS weak, own
count: no knowledge of the algebra's dimension), every field built from the data alone:
  amplitude  u d_u: u_t = F[u] homogeneous of degree 1 -- linear: of the library d_x^k (u^j), j <= 4, k <= 6, and a
             constant, only j = 1;
  boost      t d_x + c d_u, c read as the direction of the plane (t d_x, d_u) deepest in the span (with the amplitude
             gate closed): F's only u-dependence is the advection, c u u_x -- the constant, d_x^k u (k >= 1) and
             (u^2)_x remain;
  scaling    x d_x + a t d_t + (1 - a) u d_u, a-hat read as in exponent_readout.py (the plane reduced against the
             translations and the boost read), 0 < a-hat <= 6: a term of weight j (1 - a) - k balances u_t only at its
             own exponent a(j, k) = (1 + k - j) / (2 - j) ((u^2)_x at every a, the other d_x^k (u^2) at none); kept
             within TAU of a-hat. On periodic data the linear part is a translation-invariant operator homogeneous of
             degree a-hat, c_E E + c_O O, symbols -|kappa|^a-hat (even; u_xx at 2) and i sgn(kappa) |kappa|^a-hat (odd;
             H u_xx at 2, -u_xxx at 3), which replace the local linear terms; otherwise d_x^k u within TAU of a-hat.

Rows, on the evaluation data (10 trajectories, data seed 1000 + s), the authors' test functions and MSTLS throughout
(the per-term price 1 / J_REF in every row but ref):
  ref           reference weak SINDy (exp56's library and settings)
  sym           the library of the algebra FINS measured (every gate), FINS weak through the exact solver, own count;
                sym-mk the m = k span; sym-learned the learned operator's span (own count) where it exists
  sym-scal      the scaling gate alone: nonlinear terms within TAU of a-hat kept, linear part E, O (periodic data)
  sym-ref       sym with beta refined by the residual within REF_WIDTH of a-hat (periodic data)
  prof          sym's structure assumed without FINS (boost and scaling), beta minimising the selected model's
                relative residual over BETAS, refined (periodic data)
  prof-free     the reference library plus E_beta, O_beta, beta chosen likewise (periodic data; a multiplier equal to
                a local d_x^k u at integer beta left out)
Scores: the true equation (its terms and no others, linear degrees within DEG_TOL), the exponent error, the relative
error of the identified right-hand side on the clean evaluation data (periodic data), the coefficients' relative
errors.

Run:  python exp58_symmetry_library.py readouts|run|summary [--pdes ...] [--conds ...] [--seeds ...]
      python exp58_symmetry_library.py one --pde fb150 --cond n1e-2 --seed 42 --readouts <json> --out <json>
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp42_tables as e42                                 # noqa: E402
import exp49_weak as e49                                   # noqa: E402
import exp50_two_stage as e50                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp52_bo as e52                                     # noqa: E402
import exp53_ks as e53                                     # noqa: E402
import exponent_readout as er                              # noqa: E402
import fburgers as fb                                      # noqa: E402
import method_wsindy as mw                                 # noqa: E402

POLYS, MAX_DX = (0, 1, 2, 3, 4), 6
GATE = 0.5
SHARE = 1e-3                # a boost's t d_x share (normalised basis) above roundoff; below it the shift d_u
TAU = 0.2
J_REF = 31                  # MSTLS's per-term price 1 / J: the reference library plus E, O, for every row but ref
DEG_TOL = 0.1
BETAS = np.round(np.arange(0.5, 4.0 + 1e-9, 0.05), 4)
REFINE = np.round(np.arange(-0.05, 0.05 + 1e-9, 0.005), 4)
REF_WIDTH = 0.25
REF_STEP = 0.01
FB_CONDS = {"clean": 0.0, "n1e-2": 0.01}
PERIODIC = ("kdv", "bo", "ks") + fb.EQUATIONS
CELLS = {"heat": ("noise_weak_runs", e51.CONDS), "kdv": ("noise_weak_runs", e51.CONDS), "bo": ("bo_runs", e52.CONDS),
         "ks": ("ks_runs", e53.CONDS),
         **{p: ("fb_runs", FB_CONDS) for p in fb.EQUATIONS if p != "fb100"}}
ROWS = ("ref", "sym", "sym-pure", "sym-mk", "sym-learned", "sym-scal", "sym-ref", "prof", "prof-free")
DEV = {"heat": {c: f"weak_dev/weak_heat_{t}.npz" for c, t in (("clean", "n0"), ("n1e-3", "n1e-3"), ("n1e-2", "n1e-2"),
                                                               ("n3e-2", "n3e-2"), ("n5e-2", "n5e-2"),
                                                               ("n1e-1", "n1e-1"))},
       "fb125": {"clean": "fb_dev/weak_fb125_s7.npz"}, "fb175": {"clean": "fb_dev/weak_fb175_s7.npz"},
       "fb200": {"clean": "fb_dev/weak_fb200_s7.npz"},
       "fb150": {"clean": "fb_dev/weak_fb150_s7.npz", "n1e-2": "fb_dev/weak_n1e-2_fb150_s7.npz"},
       "bo": {c: f"bo_dev/weak_bo_{t}.npz" for c, t in (("clean", "n0"), ("n1e-2", "n1e-2"), ("n3e-2", "n3e-2"))},
       "kdv": {c: f"weak_dev/weak_kdv_{t}_s4.npz" for c, t in (("clean", "n0"), ("n1e-3", "n1e-3"), ("n1e-2", "n1e-2"),
                                                               ("n3e-2", "n3e-2"), ("n5e-2", "n5e-2"),
                                                               ("n1e-1", "n1e-1"))},
       "ks": {c: f"ks_dev/weak_ks_{t}.npz" for c, t in (("clean", "n0"), ("n1e-2", "n1e-2"), ("n3e-2", "n3e-2"),
                                                        ("n5e-2", "n5e-2"), ("n1e-1", "n1e-1"))}}


def truth(pde):
    """(advection coefficient of (u^2)_x, [(parity, degree, coefficient of E or O)])."""
    if pde == "heat":
        return 0.0, [("even", 2.0, e31.ExperimentConfig().alpha_heat)]
    if pde == "kdv":
        return -3.0, [("odd", 3.0, 1.0)]
    if pde == "bo":
        return -0.5, [("odd", 2.0, 1.0)]
    if pde == "ks":
        return -0.5, [("even", 2.0, -1.0), ("even", 4.0, 1.0)]
    a = fb.alpha(pde)
    return -0.5, [("even", a, fb.nu(a))]


def native(j, k):
    """Exponent a at which d_x^k (u^j) balances u_t under x d_x + a t d_t + (1 - a) u d_u (None: no a, or every a)."""
    return None if j == 2 else (1 + k - j) / (2 - j)


def even(beta):
    return lambda kap: -np.abs(kap) ** beta


def odd(beta):
    def f(kap):
        s = 1j * np.sign(kap) * np.abs(kap) ** beta
        s[-1] = 0.0                                              # rfft Nyquist (even Nx): no odd component
        return s
    return f


def structure(g, a, periodic=True, tau=TAU, b=None, local_linear=True):
    """Local terms (j, k) of the reference library kept by the gates g (linear, boost, shift, scaling); with the
    scaling at a (u-weight b; 1 - a with the boost) the linear terms d_x^k u within tau of a (none with local_linear
    False on periodic data, where E_a, O_a enter)."""
    out = []
    for j, k, _ in mw.library(POLYS, MAX_DX):
        if g["linear"] and j != 1:
            continue
        if g["shift"] and not (j == 0 or (j == 1 and k >= 1)):
            continue
        if g["boost"] and not (j == 0 or (j == 1 and k >= 1) or (j, k) == (2, 1)):
            continue
        if g["scaling"]:
            if j == 1:
                if abs(k - a) > tau or (periodic and not local_linear):
                    continue
            elif b is None:
                if (j, k) != (2, 1) and (native(j, k) is None or abs(native(j, k) - a) > tau):
                    continue
            elif abs(j * b - k - (b - a)) > tau:
                continue
        out.append((j, k))
    return out


ASSUMED = dict(linear=False, boost=True, shift=False, scaling=True)
SCALING_ONLY = dict(linear=False, boost=False, shift=False, scaling=True)


# ---------------------------------------------------------------------------------------------------------------------
# FINS's readouts: the time-scaling exponent and the Galilean boost's containment in a span

def principal(D, P):
    """Directions of span(P) (columns P [dim, m], reduced vectors) inside span(D): squared cosines (descending) and
    the coefficient vectors over P's columns, normalised in P's normalised basis."""
    Q = e42.orth(D)
    Pn = np.linalg.norm(P, axis=0)
    Po = P / Pn
    lam, V = eigh(Po.T @ Q @ (Q.T @ Po), Po.T @ Po)
    o = np.argsort(-lam)
    V = V[:, o] / np.linalg.norm(V[:, o], axis=0)
    return lam[o], V / Pn[:, None], V


def readout_file(path, pde):
    """{'own': readout, 'mk': readout} of a FINS weak file; a readout holds the scaling plane's a-hat and containment
    (exponent_readout.py, reduced against the translations and the boost read), the boost plane's (t d_x, d_u)
    containment, d_u / t d_x ratio and t d_x share, the squared cosines of the scaling space (x d_x, t d_t, u d_u)
    inside the span with its exponents (two directions: a-hat of the one without u-weight; one: a-hat and the u-weight
    b-hat), and the span's dimension. Fields from the data alone, scaled and restricted as exp51's scorer."""
    z = np.load(path, allow_pickle=True)
    u, xs, ts = z["u"], z["xs"], z["ts"]
    Nx = u.shape[1]
    keep = (np.arange(Nx) / Nx >= 0.2) & (np.arange(Nx) / Nx <= 0.8)
    w = np.array([1 / ((xs[0, 1] - xs[0, 0]) * Nx), 1 / ts[:, -1].max(), 1 / u.std()])
    X, T = np.broadcast_to(xs[:, :, None], u.shape), np.broadcast_to(ts[:, None, :], u.shape)
    one, zero = np.ones_like(u), np.zeros_like(u)

    def vec(*f):
        return e42.vectors(np.stack(f)[None], keep, w, 8)[0]

    def red(v, Q):
        return v - Q @ (Q.T @ v)

    trans = [vec(one, zero, zero), vec(zero, one, zero)]
    Qt = e42.orth(np.stack(trans))
    out = {}
    for name, m in (("mk", None), ("own", e49.own_count(z["lam"]))):
        f = e51.fins_like(z, m)
        D = e42.vectors(f, keep, w, 8)
        D = np.where(np.isfinite(D), D, 0.0)
        lb, Vb, Vbn = principal(D, np.stack([red(vec(T, zero, zero), Qt), red(vec(zero, zero, one), Qt)], 1))
        cb = float(Vb[1, 0] / Vb[0, 0]) if Vb[0, 0] != 0 else float("inf")
        share = float(abs(Vbn[0, 0]))
        rest = list(trans)
        if lb[0] >= GATE:
            rest.append(vec(T, zero, cb * one) if share >= SHARE else vec(zero, zero, one))
        Qr = e42.orth(np.stack(rest))
        a, c = er.read(D, red(vec(X, zero, u), Qr), red(vec(zero, T, -u), Qr), np.zeros((len(D[0]), 0)))
        ls, Vs, _ = principal(D, np.stack([red(vec(X, zero, zero), Qr), red(vec(zero, T, zero), Qr),
                                           red(vec(zero, zero, u), Qr)], 1))
        v1, v2 = Vs[:, 0], Vs[:, 1]
        wz = v2[2] * v1 - v1[2] * v2                     # the direction of the top two without u-weight
        out[name] = dict(a=float(a), c=float(c), lam_boost=float(lb[0]), boost_c=cb, share=share,
                         lam_scal=[float(v) for v in ls], a_lin=float(wz[1] / wz[0]) if wz[0] else None,
                         a_one=float(v1[1] / v1[0]) if v1[0] else None, b_one=float(v1[2] / v1[0]) if v1[0] else None,
                         n=int(len(f)))
    return out


def readouts(args):
    res = {}
    if args.dev:
        for pde, conds in DEV.items():
            for c, f in conds.items():
                res.setdefault(pde, {}).setdefault(c, {})["7"] = readout_file(os.path.join(args.base, f), pde)
                print(pde, c, res[pde][c]["7"], flush=True)
    else:
        for pde in args.pdes.split(","):
            sub, conds = CELLS[pde]
            for c in conds:
                for s in args.seeds.split(","):
                    d = os.path.join(args.base, sub, c)
                    r = {}
                    for tag, pre in (("weak", ""), ("weak-learned", "learned-")):
                        p = os.path.join(d, f"{tag}_{pde}_s{s}.npz")
                        if os.path.exists(p):
                            r.update({pre + k: v for k, v in readout_file(p, pde).items()})
                    res.setdefault(pde, {}).setdefault(c, {})[s] = r
                print(pde, c, {s: r["own"] for s, r in res[pde][c].items() if r}, flush=True)
    os.makedirs(os.path.dirname(args.readouts), exist_ok=True)
    with open(args.readouts, "w") as fh:
        json.dump(res, fh, indent=1)


# ---------------------------------------------------------------------------------------------------------------------
# one run: every row on one data set

def mstls_gram(A, c, bb, M, lambdas=mw.LAMBDAS, nterms=J_REF):
    """method_wsindy's MSTLS from the normal equations (A = G^T G, c = G^T b, bb = |b|^2, scaled variables): the
    selected model's relative residual, for profiles; the reported model is refitted with the reference MSTLS."""
    nn = len(c)

    def lsq(S):
        x = np.zeros(nn)
        if S.any():
            x[S] = np.linalg.lstsq(A[np.ix_(S, S)], c[S], rcond=None)[0]
        return x

    def stls(lam):
        Xi = M * W_ls
        LB, UB = lam * np.maximum(1, bnds), np.minimum(1, bnds) / lam
        small = np.zeros(nn, bool)
        while True:
            new = (np.abs(Xi) < LB) | (np.abs(Xi) > UB)
            if new.sum() == nn or (new == small).all():
                return Xi
            small = new
            Xi = np.where(small, 0.0, Xi)
            Xi[~small] = M[~small] * lsq(~small)[~small]

    def gnorm(v):
        return np.sqrt(max(float(v @ A @ v), 0.0))

    W_ls = lsq(np.ones(nn, bool))
    GW = gnorm(W_ls)
    bnds = np.sqrt(bb) / np.sqrt(np.diag(A)) * M
    loss = []
    for lam in lambdas:
        W = stls(lam) / M
        loss.append(gnorm(W - W_ls) / GW + np.count_nonzero(W) / nterms)
    W = stls(lambdas[int(np.argmin(loss))]) / M
    return float(np.sqrt(max(bb - 2 * W @ c + W @ A @ W, 0.0) / bb))


class Lib:
    """The reference library's columns and the multipliers' on one data set, cached; fits of any sub-library."""

    def __init__(self, U, x, t):
        self.w = mw.WSINDy(polys=POLYS, max_dx=MAX_DX)
        self.U = U
        self.b, self.Mlhs = self.w.prepare(U, x, t)
        self.local = [(j, k) for j, k, _ in mw.library(POLYS, MAX_DX)]
        G, M = self.w.columns(U, [(j, k, -1) for j, k in self.local])
        self.cols = {jk: (G[:, i], M[i]) for i, jk in enumerate(self.local)}
        self.G, self.Mloc = G, M
        self.A, self.c, self.bb = G.T @ G, G.T @ self.b, float(self.b @ self.b)
        self.mult = {}

    def multipliers(self, beta):
        key = round(float(beta), 6)
        if key not in self.mult:
            G, M = self.w.columns(self.U, [(1, 0, 0), (1, 0, 1)], ((even(beta), beta), (odd(beta), beta)))
            self.mult[key] = dict(E=(G[:, 0], M[0]), O=(G[:, 1], M[1]), G=G, M=M, X=self.G.T @ G, A=G.T @ G,
                                  c=G.T @ self.b)
        return self.mult[key]

    @staticmethod
    def kinds(local, beta, tol=1e-9):
        """The multipliers entering with local terms: E_beta and O_beta, but not one of the parity of a local d_x^k u
        with |k - beta| <= tol (E for k even, O for k odd)."""
        if beta is None:
            return []
        near = [k for j, k in local if j == 1 and abs(k - beta) <= tol]
        return [kd for kd, par in (("E", 0), ("O", 1)) if not any(k % 2 == par for k in near)]

    def fit(self, local, beta=None, nterms=J_REF, tol=1e-9):
        """MSTLS on the local terms [(j, k)] plus E_beta, O_beta (beta not None): (terms, residual, lambda)."""
        names = [("loc", j, k, None) for j, k in local]
        cols = [self.cols[jk] for jk in local]
        if beta is not None:
            m = self.multipliers(beta)
            for kd in self.kinds(local, beta, tol):
                names.append((kd, 1, 0, float(beta)))
                cols.append(m[kd])
        if not cols:
            return [], 1.0, float("nan")
        G = np.stack([c for c, _ in cols], 1)
        M = np.array([s for _, s in cols]) / self.Mlhs
        W, lam, resid = self.w.mstls(G, self.b, M, nterms=nterms)
        terms = [dict(kind=kd, j=j, k=k, beta=bt, c=float(c)) for (kd, j, k, bt), c in zip(names, W) if c != 0]
        return terms, float(resid), float(lam)

    def resid_gram(self, local, beta, tol=1e-9):
        """The selected model's relative residual on the local terms plus E_beta, O_beta, from the normal equations."""
        idx = [self.local.index(jk) for jk in local]
        m = self.multipliers(beta)
        mi = [{"E": 0, "O": 1}[kd] for kd in self.kinds(local, beta, tol)]
        if not idx and not mi:
            return 1.0
        A = np.block([[self.A[np.ix_(idx, idx)], m["X"][np.ix_(idx, mi)]],
                      [m["X"][np.ix_(idx, mi)].T, m["A"][np.ix_(mi, mi)]]])
        c = np.concatenate([self.c[idx], m["c"][mi]])
        M = np.concatenate([self.Mloc[idx], m["M"][mi]]) / self.Mlhs
        return mstls_gram(A, c, self.bb, M)

    def profile(self, local_of, betas, refine=REFINE, tol=1e-9):
        """beta minimising the selected model's residual (normal equations) over betas, then over beta* + refine;
        local_of(beta) gives the local terms. The model at beta* by the reference MSTLS: (terms, residual, lambda,
        beta*)."""
        best = None
        for grid in (betas, None):
            if grid is None:
                grid = best[1] + refine
                grid = grid[grid > 0]
            for b in grid:
                r = self.resid_gram(local_of(b), b, tol)
                if best is None or r < best[0]:
                    best = (r, float(b))
        terms, resid, lam = self.fit(local_of(best[1]), best[1], tol=tol)
        return terms, resid, lam, best[1]


def name_of(terms):
    out = []
    for t in terms:
        if t["kind"] == "loc":
            out.append(f"{t['c']:+.5g} d{t['k']}(u^{t['j']})")
        else:
            out.append(f"{t['c']:+.5g} {t['kind']}^{t['beta']:.3f} u")
    return " ".join(out)


def linear_atoms(terms):
    """[(parity, degree, coefficient in the E / O convention)] of the linear terms; d_x^k u = (-1)^(k/2 + 1) E_k (k
    even), (-1)^((k - 1) / 2) O_k (k odd)."""
    out = []
    for t in terms:
        if t["kind"] == "E":
            out.append(("even", t["beta"], t["c"]))
        elif t["kind"] == "O":
            out.append(("odd", t["beta"], t["c"]))
        elif t["j"] == 1 and t["k"] >= 1:
            k = t["k"]
            sgn = (-1) ** (k // 2 + 1) if k % 2 == 0 else (-1) ** ((k - 1) // 2)
            out.append(("even" if k % 2 == 0 else "odd", float(k), sgn * t["c"]))
    merged = []                     # a multiplier at an integer degree duplicates the local term of that degree
    for p, d, c in out:
        same = [i for i, (p2, d2, _) in enumerate(merged) if p2 == p and abs(d2 - d) < 1e-9]
        if same:
            merged[same[0]] = (p, merged[same[0]][1], merged[same[0]][2] + c)
        else:
            merged.append((p, d, c))
    return merged


def symbol(parity, beta, kap):
    if parity == "even":
        return -np.abs(kap) ** beta
    s = 1j * np.sign(kap) * np.abs(kap) ** beta
    if len(kap) % 2 == 0:
        s[len(kap) // 2] = 0.0
    return s


def rhs(adv, lin, other, x, u):
    """adv (u^2)_x + sum over lin of the multipliers + other local terms [(j, k, c)] on u [N, Nx, Nt], periodic."""
    kap = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    uh = np.fft.fft(u, axis=1)

    def apply(sym, fh):
        return np.real(np.fft.ifft(sym[None, :, None] * fh, axis=1))

    out = adv * apply(1j * kap, np.fft.fft(u ** 2, axis=1)) if adv else np.zeros_like(u)
    for parity, beta, c in lin:
        out += c * apply(symbol(parity, beta, kap), uh)
    for j, k, c in other:
        f = u ** j if j else np.ones_like(u)
        out += c * (apply((1j * kap) ** k, np.fft.fft(f, axis=1)) if k else f)
    return out


def score(pde, terms, x, u_clean):
    """True equation, exponent error, relative right-hand-side error on the clean data (periodic data), coefficient
    errors."""
    adv0, lin0 = truth(pde)
    advs = [t["c"] for t in terms if t["kind"] == "loc" and (t["j"], t["k"]) == (2, 1)]
    lin = linear_atoms(terms)
    other = [(t["j"], t["k"], t["c"]) for t in terms if t["kind"] == "loc" and (t["j"], t["k"]) != (2, 1)
             and not (t["j"] == 1 and t["k"] >= 1)]
    match = len(lin) == len(lin0) and all(
        any(p == p0 and abs(d - d0) <= DEG_TOL for p, d, _ in lin) for p0, d0, _ in lin0) and all(
        any(p == p0 and abs(d - d0) <= DEG_TOL for p0, d0, _ in lin0) for p, d, _ in lin)
    ok = bool(advs) == (adv0 != 0) and not other and match
    deg = [d for p, d, _ in lin if p == lin0[0][0]] if len(lin0) == 1 else []
    eF = None
    if pde in PERIODIC:
        F0 = rhs(adv0, lin0, [], x, u_clean)
        F = rhs(advs[0] if advs else 0.0, lin, other, x, u_clean)
        eF = float(np.linalg.norm(F - F0) / np.linalg.norm(F0))
    out = dict(true_eq=ok, eF=eF, beta_err=float(abs(deg[0] - lin0[0][1])) if len(deg) == 1 else None)
    if ok:
        if adv0:
            out["adv_err"] = float(abs(advs[0] - adv0) / abs(adv0))
        out["lin_err"] = [float(abs(c - c0) / abs(c0)) for p0, d0, c0 in lin0
                          for p, d, c in lin if p == p0 and abs(d - d0) <= DEG_TOL]
    return out


def gates(r):
    """The gates of a readout and the scaling's (a-hat, u-weight b-hat or None for 1 - a-hat)."""
    boost = r["lam_boost"] >= GATE and r["share"] >= SHARE
    shift = r["lam_boost"] >= GATE and r["share"] < SHARE
    n = sum(v >= GATE for v in r["lam_scal"])
    linear = n >= 2
    a, b = None, None
    if boost:
        a = r["a"] if r["c"] >= GATE else None
    elif linear:
        a = r["a_lin"]
    elif n == 1:
        a, b = r["a_one"], r["b_one"]
    scaling = a is not None and np.isfinite(a) and 0 < a <= MAX_DX
    return dict(linear=bool(linear), boost=bool(boost and not linear), shift=bool(shift and not linear),
                scaling=bool(scaling)), a, b


def one(args):
    cfg = e31.ExperimentConfig()
    ro = json.load(open(args.readouts))[args.pde][args.cond][str(args.seed)]
    a = argparse.Namespace(pde=args.pde, n=10, dseed=args.data_seed or 1000 + args.seed, noise=args.noise, thin=1)
    per = args.pde in PERIODIC
    t0 = time.time()
    u, u_clean, xs, ts, _ = e50.data(a, cfg)
    x, t = xs[0], ts[0]
    lib = Lib(u, x, t)
    print(f"[{args.pde}] {args.cond}, seed {args.seed} (data {a.dseed}): data and columns {time.time() - t0:.0f} s; "
          f"{lib.w.info['rows']} rows", flush=True)
    rows = {}

    def put(name, terms, resid, lam, beta=None, gate=None):
        rows[name] = dict(terms=terms, names=name_of(terms), resid=resid, lam=lam, beta=beta, gates=gate,
                          **score(args.pde, terms, x, u_clean))
        r = rows[name]
        print(f"  {name:12s} {'TRUE ' if r['true_eq'] else 'wrong'}"
              + (f" eF {r['eF']:.3g}" if r["eF"] is not None else "")
              + (f" beta {beta:.3f}" if beta is not None else "") + f"  u_t = {r['names']}", flush=True)

    t1 = time.time()
    put("ref", *lib.fit(lib.local, nterms=None))
    for name, key in (("sym", "own"), ("sym-mk", "mk"), ("sym-learned", "learned-own")):
        if key not in ro:
            continue
        g, ah, bh = gates(ro[key])
        gl = dict(g, a=ah, b=bh, readout=ro[key])
        mult = ah if g["scaling"] and per else None
        put(name, *lib.fit(structure(g, ah, per, b=bh), mult, tol=TAU), beta=ah if g["scaling"] else None,
            gate=gl)
        if name == "sym" and g["scaling"] and per:
            put("sym-pure", *lib.fit(structure(g, ah, b=bh, local_linear=False), ah), beta=ah, gate=gl)
            put("sym-scal", *lib.fit(structure(SCALING_ONLY, ah, b=bh), ah, tol=TAU), beta=ah, gate=gl)
            grid = np.round(ah + np.arange(-REF_WIDTH, REF_WIDTH + 1e-9, REF_STEP), 4)
            terms, resid, lam, bb_ = lib.profile(lambda bb: structure(g, bb, b=bh), grid[grid > 0], refine=np.zeros(1),
                                                 tol=TAU)
            put("sym-ref", terms, resid, lam, beta=bb_, gate=gl)
    print(f"  (sym rows {time.time() - t1:.0f} s)", flush=True)
    if per:
        t1 = time.time()
        terms, resid, lam, b = lib.profile(lambda bb: structure(ASSUMED, bb), BETAS, tol=TAU)
        put("prof", terms, resid, lam, beta=b)
        print(f"  (prof {time.time() - t1:.0f} s)", flush=True)
        t1 = time.time()
        terms, resid, lam, b = lib.profile(lambda bb: lib.local, BETAS)
        put("prof-free", terms, resid, lam, beta=b)
        print(f"  (prof-free {time.time() - t1:.0f} s)", flush=True)
    print(f"  total {time.time() - t0:.0f} s", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(pde=args.pde, cond=args.cond, seed=args.seed, data_seed=a.dseed, noise=args.noise,
                           readouts=ro, rows=rows, wsindy={k: float(v) for k, v in lib.w.info.items()}), fh, indent=1)
        print(f"  saved {args.out}")


# ---------------------------------------------------------------------------------------------------------------------
# jobs and summary

def noise_of(pde, cond):
    return CELLS[pde][1][cond]


def jobs(args):
    out = []
    for pde in args.pdes.split(","):
        for c in args.conds.split(","):
            if c not in CELLS[pde][1]:
                continue
            for s in args.seeds.split(","):
                f = os.path.join(args.runs, c, f"sym_{pde}_s{s}.json")
                out.append((f, ["exp58_symmetry_library.py", "one", "--pde", pde, "--cond", c, "--seed", s,
                                "--noise", f"{noise_of(pde, c):g}", "--readouts", args.readouts, "--out", f]))
    return out


def summary(args):
    res, lines = {}, ["# Symmetry-informed weak SINDy (exp58): true equation (count), median relative right-hand-side "
                      "error on the clean data, median |beta - alpha| (rows with one linear term)\n"]
    for pde in args.pdes.split(","):
        for c in CELLS[pde][1]:
            fs = [os.path.join(args.runs, c, f"sym_{pde}_s{s}.json") for s in args.seeds.split(",")]
            js = [json.load(open(f)) for f in fs if os.path.exists(f)]
            if not js:
                continue
            cell = res.setdefault(pde, {}).setdefault(c, {})
            for row in ROWS:
                rs = {str(j["seed"]): j["rows"][row] for j in js if row in j["rows"]}
                if not rs:
                    continue
                cell[row] = rs
    for pde, cs in res.items():
        lines.append(f"## {pde}\n")
        lines.append("| row | " + " | ".join(cs) + " |")
        lines.append("|---" * (len(cs) + 1) + "|")
        for row in ROWS:
            cells = []
            for c in cs:
                rs = cs[c].get(row)
                if not rs:
                    cells.append("")
                    continue
                n = len(rs)
                ok = sum(r["true_eq"] for r in rs.values())
                eF = [r["eF"] for r in rs.values() if r["eF"] is not None]
                be = [r["beta_err"] for r in rs.values() if r["beta_err"] is not None]
                cells.append(f"{ok}/{n}" + (f", eF {np.median(eF):.2g}" if eF else "")
                             + (f", dβ {np.median(be):.3f}" if be else ""))
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        lines.append("")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["readouts", "one", "run", "summary"])
    ap.add_argument("--pde", default=None)
    ap.add_argument("--cond", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--pdes", default=",".join(CELLS))
    ap.add_argument("--conds", default="clean,n1e-3,n1e-2,n3e-2,n5e-2,n1e-1")
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--dev", action="store_true", help="readouts of the development runs (seed 7, data 1007)")
    base = os.environ.get("FINS_RESULTS_DIR", "results")
    ap.add_argument("--base", default=base, help="directory of the FINS runs")
    ap.add_argument("--readouts", default=os.path.join(base, "sym_runs", "readouts.json"))
    ap.add_argument("--runs", default=os.path.join(base, "sym_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.what == "readouts":
        readouts(args)
    elif args.what == "one":
        one(args)
    elif args.what == "run":
        e51.run(args, jobs(args))
    else:
        args.out = args.out or os.path.join(e51.HERE, "results", "symlib")
        summary(args)


if __name__ == "__main__":
    main()
