"""Checks of pde/method_kreider.py against the authors' MATLAB code (Kreider, Harlim, Huang, arXiv 2511.09779) and on
analytic surfaces.

  rows_3c, rows_3d   rows S . pr v of the affine class equal local_row of Fig_3c.m (first jet space) and Fig_3d.m
                     (second jet space), transcribed below, for random inputs; also through a degree-2 class and a
                     normalised class (change of basis), and A = sum p'p against their accumulation
  prolongation       pr^(4) v of a random degree-2 normalised generator against D_J Q computed by differentiating
                     Q(t, x, u(t, x), u_t, u_x) as a composite function of (t, x) (independent of the total derivative)
  exact_invariance   with exact jets and normals (p = 1..4): rows of the known invariances of sin(t + x) and of the
                     heat kernel vanish; numerical null-space dimension of the affine class
  reference_port     a literal per-point port of the Fig_3c.m / Fig_3d.m loops against the vectorised pipeline
  plane              jets of an affine graph are exact (SVD and GMLS exact, derivatives exact up to order 3)
  analytic_surface   tangent planes and derivatives up to order 3 of a known surface: GMLS against SVD alone,
                     convergence in n
  pooled             codimension-1 and -2 frames of point clouds in R^4 and R^5 (tangent dimension 3) against the
                     exact normals

Run:  python kreider_check.py [--out DIR]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy.linalg import null_space
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import method_kreider as mk                                # noqa: E402

OUT = os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "kreider_dev", "validation")
TRUTH_3C = [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0]
TRUTH_3D = [0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1]
T_SYM, X_SYM = sp.symbols("t x")
U_TRANSPORT = sp.sin(T_SYM + X_SYM)
U_HEAT = sp.exp(-X_SYM ** 2 / (4 * T_SYM)) / sp.sqrt(4 * sp.pi * T_SYM)


class Log:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(v) for v in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def local_row_3c(X, S):
    """local_row of Fig_3c.m (1-based indices shifted)."""
    p = np.zeros(12)
    p[0] = S[0]
    p[1] = S[0] * X[0] - S[3] * X[3]
    p[2] = S[0] * X[1] - S[4] * X[3]
    p[3] = S[0] * X[2] - S[3] * X[3] ** 2 - S[4] * X[3] * X[4]
    p[4] = S[1]
    p[5] = S[1] * X[0] - S[3] * X[4]
    p[6] = S[1] * X[1] - S[4] * X[4]
    p[7] = S[1] * X[2] - S[4] * X[4] ** 2 - S[3] * X[3] * X[4]
    p[8] = S[2]
    p[9] = S[3] + S[2] * X[0]
    p[10] = S[4] + S[2] * X[1]
    p[11] = S[2] * X[2] + S[3] * X[3] + S[4] * X[4]
    return p


def local_row_3d(X, S):
    """local_row of Fig_3d.m (1-based indices shifted)."""
    p = np.zeros(12)
    p[0] = S[0]
    p[1] = S[0] * X[0] - S[3] * X[3] - 2 * S[5] * X[5] - S[6] * X[6]
    p[2] = S[0] * X[1] - S[6] * X[5] - 2 * S[7] * X[6] - S[4] * X[3]
    p[3] = (S[0] * X[2] - S[3] * X[3] ** 2 - S[6] * (2 * X[3] * X[6] + X[5] * X[4])
            - S[7] * (X[3] * X[7] + 2 * X[6] * X[4]) - 3 * S[5] * X[3] * X[5] - S[4] * X[3] * X[4])
    p[4] = S[1]
    p[5] = S[1] * X[0] - 2 * S[5] * X[6] - S[3] * X[4] - S[6] * X[7]
    p[6] = S[1] * X[1] - S[4] * X[4] - 2 * S[7] * X[7] - S[6] * X[6]
    p[7] = (S[1] * X[2] - S[4] * X[4] ** 2 - S[5] * (2 * X[3] * X[6] + X[5] * X[4])
            - S[6] * (X[3] * X[7] + 2 * X[6] * X[4]) - S[3] * X[3] * X[4] - 3 * S[7] * X[4] * X[7])
    p[8] = S[2]
    p[9] = S[3] + S[2] * X[0]
    p[10] = S[4] + S[2] * X[1]
    p[11] = S[2] * X[2] + S[3] * X[3] + S[5] * X[5] + S[6] * X[6] + S[4] * X[4] + S[7] * X[7]
    return p


def authors_map(gens):
    """C [n_basis, 12]: column i holds the class coefficients of the authors' i-th affine basis generator."""
    return np.stack([gens.from_authors_affine(np.eye(12)[i]) for i in range(12)], 1)


def check_rows(log, p, local_row, rng, n=400):
    """Rows of the affine, a degree-2 and a normalised class against the transcribed local_row."""
    D = mk.jet_dim(p)
    Z = rng.normal(size=(n, D)) * rng.uniform(0.2, 3.0, size=D)
    S = rng.normal(size=(n, D, D - 2))
    theirs = np.stack([[local_row(Z[i], S[i, :, j]) for j in range(D - 2)] for i in range(n)])
    A_theirs = sum(np.outer(r, r) for r in theirs.reshape(-1, 12))
    out = {}
    for name, g in (("affine", mk.PolynomialGenerators(p, 1)), ("degree2", mk.PolynomialGenerators(p, 2)),
                    ("normalised", mk.PolynomialGenerators(p, 2, shift=(0.3, -0.7, 0.1), scale=(2.0, 0.5, 3.0)))):
        C = authors_map(g)
        mine = g.rows(Z, S) @ C
        rel = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
        A_mine = C.T @ g.gram(Z, S, chunk=97) @ C
        relA = float(np.linalg.norm(A_mine - A_theirs) / np.linalg.norm(A_theirs))
        perm = None
        if name == "affine":
            perm = [int(np.argmax(np.abs(C[:, i]))) for i in range(12)]
        out[name] = dict(max_rel_row=rel, rel_A=relA, perm=perm)
        log(f"  {name:10s} max |rows - local_row| / max |local_row| = {rel:.2e}   "
            f"|A - A_theirs| / |A_theirs| = {relA:.2e}")
    ok = all(v["max_rel_row"] < 1e-13 and v["rel_A"] < 1e-13 for v in out.values())
    ok = ok and out["affine"]["perm"] == list(range(12))
    return ok, out


def exact_jets(fexpr, p, T, X):
    """Exact jet coordinates [n, jet_dim(p)] of u = fexpr(t, x)."""
    cols = [T, X]
    for a, b in mk.multi_indices(p):
        cols.append(_eval(sp.diff(fexpr, T_SYM, a, X_SYM, b), T, X))
    return np.stack(cols, 1)


def exact_frames(fexpr, p, T, X):
    """Exact tangent [n, D, 2] and orthonormal normal [n, D, D - 2] frames of the order-p prolonged graph."""
    Tt = [np.ones_like(T), np.zeros_like(T)] + [_eval(sp.diff(fexpr, T_SYM, a + 1, X_SYM, b), T, X)
                                                for a, b in mk.multi_indices(p)]
    Tx = [np.zeros_like(T), np.ones_like(T)] + [_eval(sp.diff(fexpr, T_SYM, a, X_SYM, b + 1), T, X)
                                                for a, b in mk.multi_indices(p)]
    Tg = np.stack([np.stack(Tt, 1), np.stack(Tx, 1)], 2)
    return Tg, mk._complement(Tg)


def _eval(expr, T, X):
    return np.broadcast_to(sp.lambdify((T_SYM, X_SYM), expr, "numpy")(T, X), T.shape).astype(float)


def check_prolongation(log, rng, n=60, p=4):
    """pr^(p) of a random normalised degree-2 generator against the composite-function derivatives."""
    g = mk.PolynomialGenerators(p, 2, shift=(0.5, -0.2, 0.3), scale=(1.5, 0.8, 2.0))
    c = rng.normal(size=g.n_basis)
    fexpr = sp.exp(sp.Rational(3, 10) * T_SYM) * sp.sin(X_SYM) + sp.Rational(1, 5) * T_SYM ** 2 * X_SYM ** 3
    tau, chi, ups = g._normalised(T_SYM, X_SYM, fexpr)
    comps = [sum(sp.Float(c[k * g.n_mono + i]) * tau ** e[0] * chi ** e[1] * ups ** e[2]
                 for i, e in enumerate(g.monomials)) for k in range(3)]
    Q = comps[2] - comps[0] * sp.diff(fexpr, T_SYM) - comps[1] * sp.diff(fexpr, X_SYM)
    P = rng.uniform(0.2, 1.0, size=(n, 2))
    T, X = P[:, 0], P[:, 1]
    ref = [_eval(e, T, X) for e in comps]
    for a, b in mk.multi_indices(p)[1:]:
        e = (sp.diff(Q, T_SYM, a, X_SYM, b) + comps[0] * sp.diff(fexpr, T_SYM, a + 1, X_SYM, b)
             + comps[1] * sp.diff(fexpr, T_SYM, a, X_SYM, b + 1))
        ref.append(_eval(e, T, X))
    ref = np.stack(ref, 1)
    mine = g.prolongation(exact_jets(fexpr, p, T, X)) @ c
    rel = float(np.abs(mine - ref).max() / np.abs(ref).max())
    log(f"  p = {p}, degree 2, normalised: max |pr v - composite| / max |composite| = {rel:.2e} "
        f"(D = {g.dim}, {g.n_basis} basis generators)")
    return rel < 1e-11, dict(max_rel=rel)


def check_exact_invariance(log, rng, n=4000):
    """Known invariances annihilate the rows on exact normals; null-space dimension of the affine class."""
    cases = {"transport": (U_TRANSPORT, (-0.5, 0.5), [("1", "-1", "0"), ("t", "-t", "0"), ("x", "-x", "0"),
                                                     ("u", "-u", "0")]),
             "heat": (U_HEAT, (1.0, 2.0), [("2*t", "x", "-u")])}
    proj = ("4*t**2", "4*t*x", "-(x**2 + 2*t)*u")
    out, ok = {}, True
    for name, (fexpr, (a, b), gens_true) in cases.items():
        P = rng.uniform(a, b, size=(n, 2))
        for p in range(1, 5):
            Z = exact_jets(fexpr, p, P[:, 0], P[:, 1])
            _, S = exact_frames(fexpr, p, P[:, 0], P[:, 1])
            g = mk.PolynomialGenerators(p, 1)
            R = g.rows(Z, S)
            scale = np.abs(R).max()
            res = max(float(np.abs(R @ g.coefficients(*v)).max() / scale) for v in gens_true)
            sv = np.linalg.svd(g.gram(Z, S), compute_uv=False)
            dim = int((sv < 1e-12 * sv[0]).sum())
            rec = dict(max_rel_residual=res, null_dim=dim, sv_rel=(sv / sv[0]).tolist())
            line = f"  {name:9s} p = {p}: max rel residual of the known invariances {res:.1e}, affine null dim {dim}"
            ok = ok and res < 1e-12 and dim == len(gens_true)
            if name == "heat":
                g3 = mk.PolynomialGenerators(p, 3)
                R3 = g3.rows(Z, S)
                res3 = float(np.abs(R3 @ g3.coefficients(*proj)).max() / np.abs(R3).max())
                rec["projective_residual_degree3"] = res3
                line += f", projective (degree 3) {res3:.1e}"
                ok = ok and res3 < 1e-12
            log(line + f", sv/sv_max below the null: {np.array2string(sv[-dim - 2:] / sv[0], precision=2)}")
            out[f"{name}_p{p}"] = rec
    return ok, out


def ref_pipeline(X3, p, K, ell, J):
    """Literal per-point port of the Fig_3c.m (p = 1) / Fig_3d.m (p = 2) loops: knnsearch, svd(M), null, the
    Vandermonde loop, V \\ x_proj, the tangent update; returns the jets X and the final normals [n, D, D - 2]."""
    X = np.array(X3, dtype=float)
    for level in range(p + 1):
        n, D = X.shape
        idx = cKDTree(X).query(X, k=K)[1]
        T1, T2, S = np.zeros((D, n)), np.zeros((D, n)), np.zeros((D, D - 2, n))
        for i in range(n):
            M = X[idx[i]].T - X[i][:, None]
            U = np.linalg.svd(M)[0]
            T1[:, i], T2[:, i] = U[:, 0], U[:, 1]
            S[:, :, i] = null_space(np.stack([T1[:, i], T2[:, i]]))
            for _ in range(J):
                x_proj = (M.T @ np.column_stack([T1[:, i], T2[:, i], S[:, :, i]])).T
                cols = []
                for d in range(1, ell + 1):
                    for pp in range(d + 1):
                        qq = d - pp
                        cols.append(x_proj[0, 1:] ** pp * x_proj[1, 1:] ** qq)
                V = np.column_stack(cols)
                a1 = [np.linalg.lstsq(V, x_proj[2 + k, 1:], rcond=None)[0][:2] for k in range(D - 2)]
                t_new1 = T1[:, i] + sum(a1[k][1] * S[:, k, i] for k in range(D - 2))
                t_new2 = T2[:, i] + sum(a1[k][0] * S[:, k, i] for k in range(D - 2))
                T1[:, i] = t_new1 / np.linalg.norm(t_new1)
                T2[:, i] = t_new2 / np.linalg.norm(t_new2)
                S[:, :, i] = null_space(np.stack([T1[:, i], T2[:, i]]))
        if level == p:
            return X, np.moveaxis(S, 2, 0)
        new = []
        for i in range(n):
            A = np.stack([T1[:2, i], T2[:2, i]])
            cols_k = list(range(D - level - 1, D))
            du = [np.linalg.solve(A, np.array([T1[c, i], T2[c, i]])) for c in cols_k]
            new.append([du[0][0]] + [d[1] for d in du])
        X = np.column_stack([X, np.array(new)])


def check_reference_port(log, rng, N=24):
    """Vectorised pipeline against the literal port, on the Fig 3c and 3d data."""
    out, ok = {}, True
    for fig, (a, b, K, ell, p, fexpr, truth, col) in {
            "3c": (-0.5, 0.5, 20, 3, 1, U_TRANSPORT, TRUTH_3C, -2),
            "3d": (1.0, 2.0, 40, 4, 2, U_HEAT, TRUTH_3D, -1)}.items():
        P = rng.uniform(a, b, size=(N * N, 2))
        u = _eval(fexpr, P[:, 0], P[:, 1])
        t0 = time.time()
        Xr, Sr = ref_pipeline(np.column_stack([P, u]), p, K, ell, 2)
        t_ref = time.time() - t0
        g = mk.PolynomialGenerators(p, 1)
        t0 = time.time()
        r = mk.discover(P[:, 0], P[:, 1], u, g, K, ell, 2, chunk=101)
        t_vec = time.time() - t0
        A_ref = sum(np.outer(rr, rr) for i in range(len(Xr)) for rr in
                    [(local_row_3c if p == 1 else local_row_3d)(Xr[i], Sr[i, :, j]) for j in range(Sr.shape[2])])
        sv_ref, V_ref = mk.null_space_svd(A_ref)
        dZ = float(np.abs(r["Z"] - Xr).max() / np.abs(Xr).max(axis=0).max())
        dP = float(np.abs(np.einsum("nik,njk->nij", r["S"], r["S"]) - np.einsum("nik,njk->nij", Sr, Sr)).max())
        dA = float(np.linalg.norm(r["A"] - A_ref) / np.linalg.norm(A_ref))
        e_ref = mk.sin_subspace(np.array(truth, float), V_ref[:, col])
        e_vec = mk.sin_subspace(np.array(truth, float), r["V"][:, col])
        out[fig] = dict(n=N * N, max_rel_jet=dZ, max_normal_projector=dP, rel_A=dA, err_ref=e_ref, err_vec=e_vec,
                        t_ref=t_ref, t_vec=t_vec)
        log(f"  Fig {fig}, n = {N * N}: jets {dZ:.1e}, normal projectors {dP:.1e}, A {dA:.1e}; "
            f"error ref {e_ref:.6e} vec {e_vec:.6e}; time ref {t_ref:.1f} s vec {t_vec:.2f} s")
        ok = ok and dZ < 1e-9 and dP < 1e-9 and dA < 1e-9 and abs(e_ref - e_vec) < 1e-6 * max(e_ref, 1e-3)
    return ok, out


def check_plane(log, rng, n=1600):
    """Affine graph u = 1 + 2t - 3x: jets up to order 3 and normals exact up to round-off, which each jet level
    amplifies (differences over neighbourhoods of radius h, about 1 / h per level)."""
    P = rng.uniform(0.0, 1.0, size=(n, 2))
    Z, T, S = mk.jet_frames(P[:, 0], P[:, 1], 1 + 2 * P[:, 0] - 3 * P[:, 1], 3, 20, 3, 2)
    truth = np.zeros_like(Z)
    truth[:, :2] = P
    truth[:, 2] = 1 + 2 * P[:, 0] - 3 * P[:, 1]
    truth[:, 3], truth[:, 4] = 2.0, -3.0
    by_order = [float(np.abs(Z - truth)[:, [2 + i for i, J in enumerate(mk.multi_indices(3)) if sum(J) == k]].max())
                for k in range(4)]
    Tg, _ = exact_frames(1 + 2 * T_SYM - 3 * X_SYM, 3, P[:, 0], P[:, 1])
    e_n = float(np.abs(np.einsum("nik,nil->nkl", S, Tg)).max())
    log(f"  max jet error by order 0..3 {np.array2string(np.array(by_order), precision=1)}, "
        f"max |S' T_exact| at order 3 {e_n:.1e}")
    return max(by_order) < 1e-6 and e_n < 1e-6, dict(max_jet_error_by_order=by_order, max_normal_tangent=e_n)


def principal_sin(S, S_true):
    """Per-point sin of the largest principal angle between the column spans [n, D, q] of S and S_true."""
    Qt = np.linalg.qr(S_true)[0]
    R = S - Qt @ (np.swapaxes(Qt, 1, 2) @ S)
    return np.linalg.norm(R, ord=2, axis=(1, 2))


def check_analytic_surface(log, rng):
    """u = sin(t) cos(2x) + t x / 2 on [0, 1]^2, p = 3, K = 40, ell = 4, J = 2 and J = 0; interior points (0.15 from
    the boundary)."""
    fexpr = sp.sin(T_SYM) * sp.cos(2 * X_SYM) + T_SYM * X_SYM / 2
    labels = mk.jet_labels(3)
    out, ok = {}, True
    med_prev = None
    for N in (80, 160):
        P = rng.uniform(0.0, 1.0, size=(N * N, 2))
        inner = np.all((P > 0.15) & (P < 0.85), axis=1)
        u = _eval(fexpr, P[:, 0], P[:, 1])
        Tg0, S0 = exact_frames(fexpr, 0, P[:, 0], P[:, 1])
        Y0 = np.column_stack([P, u])
        ang = {J: float(np.median(principal_sin(mk.local_frames(Y0, 2, 40, 4, J)[1][inner], S0[inner])))
               for J in (0, 2)}
        Z, _, S = mk.jet_frames(P[:, 0], P[:, 1], u, 3, 40, 4, 2)
        Ze = exact_jets(fexpr, 3, P[:, 0], P[:, 1])
        _, Se = exact_frames(fexpr, 3, P[:, 0], P[:, 1])
        med = np.median(np.abs(Z - Ze)[inner], axis=0)[3:]
        scale = np.median(np.abs(Ze)[inner], axis=0)[3:]
        ang3 = float(np.median(principal_sin(S[inner], Se[inner])))
        out[f"N{N}"] = dict(n=N * N, tangent_sin_svd=ang[0], tangent_sin_gmls=ang[2], normal_sin_order3=ang3,
                            median_abs_err={k: float(v) for k, v in zip(labels[3:], med)},
                            median_abs_value={k: float(v) for k, v in zip(labels[3:], scale)})
        log(f"  n = {N}^2: level-0 tangent sin angle SVD {ang[0]:.1e}, GMLS {ang[2]:.1e}; order-3 normal space "
            f"{ang3:.1e}")
        log("    median |error| " + ", ".join(f"{k} {v:.1e}" for k, v in zip(labels[3:], med)))
        ok = ok and ang[2] < ang[0] / 10 and np.all(med[:2] < 1e-4 * scale[:2])
        if med_prev is not None:
            ok = ok and np.all(med < med_prev)
        med_prev = med
    return bool(ok), out


def check_pooled(log, rng, n=20000):
    """Codimension 1 in R^4 (y4 = F1(y1, y2, y3)) and codimension 2 in R^5 (y4 = F1, y5 = F2), K = 60, ell = 3."""
    Y3 = rng.uniform(-1.0, 1.0, size=(n, 3))
    y1, y2, y3 = Y3.T
    F1 = np.sin(y1) + y2 * y3 + 0.5 * y3 ** 2
    G1 = np.column_stack([np.cos(y1), y3, y2 + y3, -np.ones(n), np.zeros(n)])
    F2 = np.exp(0.5 * y1) * y3 - y2 ** 2
    G2 = np.column_stack([0.5 * np.exp(0.5 * y1) * y3, -2 * y2, np.exp(0.5 * y1), np.zeros(n), -np.ones(n)])
    inner = np.all(np.abs(Y3) < 0.8, axis=1)
    out, ok = {}, True
    for name, Y, Nt in (("codim1_R4", np.column_stack([Y3, F1]), G1[:, :4, None]),
                        ("codim2_R5", np.column_stack([Y3, F1, F2]), np.stack([G1, G2], 2))):
        q = Nt.shape[2]
        rec = {}
        for J in (0, 2):
            t0 = time.time()
            T, S = mk.pooled_normals(Y, q, 60, 3, J)
            rec[f"J{J}"] = dict(median_sin=float(np.median(principal_sin(S[inner], Nt[inner]))),
                                p90_sin=float(np.quantile(principal_sin(S[inner], Nt[inner]), 0.9)),
                                orth=float(np.abs(np.einsum("nik,nil->nkl", S, T)).max()), time=time.time() - t0)
        out[name] = rec
        log(f"  {name}: median sin angle to the exact normals SVD {rec['J0']['median_sin']:.1e}, "
            f"GMLS {rec['J2']['median_sin']:.1e} (p90 {rec['J2']['p90_sin']:.1e}); max |S' T| {rec['J2']['orth']:.1e}; "
            f"{rec['J2']['time']:.1f} s")
        ok = ok and rec["J2"]["median_sin"] < rec["J0"]["median_sin"] / 10 and rec["J2"]["median_sin"] < 1e-3
        ok = ok and rec["J2"]["orth"] < 1e-12
    return bool(ok), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = Log(os.path.join(args.out, f"kreider_check_{stamp}.log"))
    log(f"kreider_check {stamp}  numpy {np.__version__}  sympy {sp.__version__}")
    tests = [("rows_3c", lambda r: check_rows(log, 1, local_row_3c, r)),
             ("rows_3d", lambda r: check_rows(log, 2, local_row_3d, r)),
             ("prolongation", lambda r: check_prolongation(log, r)),
             ("exact_invariance", lambda r: check_exact_invariance(log, r)),
             ("reference_port", lambda r: check_reference_port(log, r)),
             ("plane", lambda r: check_plane(log, r)),
             ("analytic_surface", lambda r: check_analytic_surface(log, r)),
             ("pooled", lambda r: check_pooled(log, r))]
    res = {}
    for i, (name, fn) in enumerate(tests):
        log(f"[{name}]")
        t0 = time.time()
        ok, info = fn(np.random.default_rng(1000 + i))
        res[name] = dict(passed=bool(ok), seconds=round(time.time() - t0, 2), **info)
        log(f"  -> {'PASS' if ok else 'FAIL'} ({res[name]['seconds']} s)")
    with open(os.path.join(args.out, "kreider_check.json"), "w") as f:
        json.dump(dict(stamp=stamp, results=res), f, indent=1)
    n_pass = sum(v["passed"] for v in res.values())
    log(f"{n_pass}/{len(res)} passed")
    sys.exit(0 if n_pass == len(res) else 1)


if __name__ == "__main__":
    main()
