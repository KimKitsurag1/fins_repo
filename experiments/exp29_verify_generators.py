"""Experiment 29 -- are FINS's generators symmetries of the TRUE equation?

The span test showed that FINS's recovered span contains the direction along
which the Burgers algebra leaves the affine class (xi ~ x t). That is a
statement about a subspace, not about generators. The only non-affine
Burgers symmetry is projective (x t, t^2, x - t u), and the determining
equations bind its components: x t in xi must come with t^2 in eta and
x - t u in phi. A field carrying x t without those partners is a fragment,
not a symmetry. This script measures, for each FINS generator in an
artifact, three things that do not depend on the surrogate it was trained
through:

  local residual  the determining equation Q_t = L_u[Q] on the data, local
                  x-derivatives, interior only -- the criterion that sees
                  scaling and projective on a torus
  flow residual   ||dS[u0](Q0) - Q|| / ||Q|| through the solver; valid only
                  for generators compatible with the periodic problem
  symbolic fit    least squares of xi, eta, phi on monomials
                  {1, x, t, u, xt, x^2, t^2, tu, xu} across all reference
                  samples: a readable formula and the share it explains
  algebra split   projection onto span(GT): which combination of true
                  symmetries the generator is, and how much lies outside

Controls on the same trajectories: every GT generator (the attainable floor
-- on a periodic grid at Nx = 64 even the true projective passes only with
a window, exp25), projective with its partners removed, and random smooth
fields (the chance level). The combination inside FINS's span that best
aligns with the non-affine direction is tested as well: it is the object
the span test credited.

Run:  python exp29_verify_generators.py --artifact generators_burgers_v3_seed42.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pde"))
from config import ExperimentConfig                     # noqa: E402
from data_generation import PDESolver                    # noqa: E402
from ground_truth import GroundTruthAlgebra               # noqa: E402

from determining import determining_matrix, gt_coefficients  # noqa: E402

MONO = ("1", "x", "t", "u", "xt", "x2", "t2", "tu", "xu")


def monomials(X, T, U):
    return np.stack([np.ones_like(X), X, T, U, X * T, X ** 2, T ** 2,
                     T * U, X * U], axis=-1)


def dx_spec(u, L):
    k = 2 * np.pi * np.fft.fftfreq(u.shape[-2], d=L / u.shape[-2])
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(u, axis=-2),
                               axis=-2))


def dt_fd(u, dt):
    out = np.empty_like(u)
    out[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    out[..., 0] = (u[..., 1] - u[..., 0]) / dt
    out[..., -1] = (u[..., -1] - u[..., -2]) / dt
    return out


def exact_residuals(gens, u_ref, x, t, nu, margin=0.1, t_margin=2):
    """Windowed ||Q||-normalised residual of each generator, per sample.

    gens: list of (xi, eta, phi), each [n_ref, Nx, Nt]. All (generator,
    sample, sign) perturbations go through ONE batched solve.
    """
    L = x[-1] + (x[1] - x[0])
    dt = t[1] - t[0]
    T, Nt = t[-1], len(t)
    n_ref = u_ref.shape[0]
    ux, ut = dx_spec(u_ref, L), dt_fd(u_ref, dt)
    Qs = [phi - xi * ux - eta * ut for xi, eta, phi in gens]
    u0 = u_ref[:, :, 0]
    eps = [1e-4 * np.linalg.norm(u0, axis=1, keepdims=True)
           / (np.linalg.norm(Q[:, :, 0], axis=1, keepdims=True) + 1e-30)
           for Q in Qs]
    batch = np.concatenate([np.concatenate([u0 + e * Q[:, :, 0],
                                            u0 - e * Q[:, :, 0]])
                            for Q, e in zip(Qs, eps)])
    sol = np.asarray(PDESolver.solve_burgers(batch, nu, T, Nt, verbose=False))
    keep = (x >= margin * L) & (x <= (1 - margin) * L)
    tsl = slice(t_margin, Nt - t_margin)
    out = []
    for g, (Q, e) in enumerate(zip(Qs, eps)):
        blk = sol[g * 2 * n_ref:(g + 1) * 2 * n_ref]
        dS = (blk[:n_ref] - blk[n_ref:]) / (2 * e[:, :, None])
        r = (dS - Q)[:, keep][:, :, tsl]
        q = Q[:, keep][:, :, tsl]
        out.append(np.linalg.norm(r.reshape(n_ref, -1), axis=1)
                   / (np.linalg.norm(q.reshape(n_ref, -1), axis=1) + 1e-30))
    return out


def local_residuals(gens, u_ref, x, t, a, b, margin=0.1, t_margin=3):
    """Local determining-equation residual, per sample.

    A symmetry's characteristic obeys the linearised equation
    Q_t = L_u[Q], L_u[Q] = -a (Q u_x + u Q_x) + b Q_xx. x-derivatives of Q
    are LOCAL (no periodic wrap) and only interior points are scored, so the
    seam of x-weighted generators never enters. This is the right check for
    scaling and projective on a torus: they are symmetries of the equation
    but not of the periodic problem, so a flow-map test (exact_residuals)
    cannot confirm them once transport carries the seam inward. Uses the
    equation (a, b) -- a verification, not a discovery, criterion.
    """
    L = x[-1] + (x[1] - x[0])
    k = 2 * np.pi * np.fft.fftfreq(u_ref.shape[1], d=L / u_ref.shape[1])
    spec = lambda f, n: np.real(np.fft.ifft((1j * k[:, None]) ** n
                                            * np.fft.fft(f, axis=1), axis=1))
    gx = lambda f: np.gradient(f, x[1] - x[0], axis=1, edge_order=2)
    gt = lambda f: np.gradient(f, t[1] - t[0], axis=2, edge_order=2)
    ux, uxx = spec(u_ref, 1), spec(u_ref, 2)
    ut = -a * u_ref * ux + b * uxx
    keep = (x >= margin * L) & (x <= (1 - margin) * L)
    tsl = slice(t_margin, len(t) - t_margin)
    out = []
    for xi, eta, phi in gens:
        Q = phi - xi * ux - eta * ut
        Qx = gx(Q)
        Lq = -a * (Q * ux + u_ref * Qx) + b * gx(Qx)
        Qt = gt(Q)
        rs = []
        for r in range(u_ref.shape[0]):
            d = (Qt[r] - Lq[r])[keep][:, tsl]
            s1, s2 = Qt[r][keep][:, tsl], Lq[r][keep][:, tsl]
            rs.append(np.linalg.norm(d) / np.sqrt(np.linalg.norm(s1) ** 2
                                                  + np.linalg.norm(s2) ** 2
                                                  + 1e-30))
        out.append(np.array(rs))
    return out


def shoot_residuals(gens, u_ref, x, t, nu, K=16, every=16, seam=0.2):
    """The training loss (multiple-shooting increment form) through the EXACT
    solver, in training units: per sample sum_j ||a - b||^2 / sum_j (||a||^2 +
    ||b||^2), averaged. A slot that is low here is a symmetry the loss
    genuinely admits; low through the surrogate but high here, a spurious
    near-symmetry of the surrogate."""
    L = x[-1] + (x[1] - x[0])
    Nt = len(t)
    k = 2 * np.pi * np.fft.fftfreq(u_ref.shape[1], d=L / u_ref.shape[1])
    ux = np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(u_ref, axis=1), axis=1))
    ut = dt_fd(u_ref, t[1] - t[0])
    keep = (x >= seam * L) & (x <= (1 - seam) * L)
    starts = list(range(0, Nt - K, every))
    R = u_ref.shape[0]
    Qs = [phi - xi * ux - eta * ut for xi, eta, phi in gens]
    u0 = np.concatenate([u_ref[:, :, j] for j in starts])
    batch, epss = [], []
    for Q in Qs:
        q0 = np.concatenate([Q[:, :, j] for j in starts])
        qn = np.linalg.norm(q0, axis=1, keepdims=True)
        e = np.where(qn > 1e-12, 1e-4 * np.linalg.norm(u0, axis=1, keepdims=True)
                     / np.maximum(qn, 1e-30), 0.0)
        epss.append(e)
        batch += [u0 + e * q0, u0 - e * q0]
    sol = np.asarray(PDESolver.solve_burgers(np.concatenate(batch), nu, t[K],
                                             K + 1, verbose=False))
    n = u0.shape[0]
    out = []
    for g, (Q, e) in enumerate(zip(Qs, epss)):
        blk = sol[g * 2 * n:(g + 1) * 2 * n]
        ee = np.where(e > 0, e, 1.0)[:, :, None]
        dS = np.where(e[:, :, None] > 0, (blk[:n] - blk[n:]) / (2 * ee), 0.0)
        num, den = np.zeros(R), np.zeros(R)
        for m, j in enumerate(starts):
            d = dS[m * R:(m + 1) * R]
            a = (d[:, :, 1:K + 1] - Q[:, :, j:j + 1])[:, keep]
            b = (Q[:, :, j + 1:j + K + 1] - Q[:, :, j:j + 1])[:, keep]
            num += ((a - b) ** 2).sum(axis=(1, 2))
            den += (a ** 2).sum(axis=(1, 2)) + (b ** 2).sum(axis=(1, 2))
        out.append(float(np.mean(num / (den + 1e-30))))
    return out


def region(crop):
    """Index region for [R, Nx, Nt] fields: an int crops every edge; a pair
    (x_slice, t_slice) is used as given."""
    if isinstance(crop, tuple):
        return (slice(None),) + crop
    return (slice(None), slice(crop, -crop), slice(crop, -crop))


def nonlocality(gen, u_ref, crop, delta_frac=0.02):
    """Point-symmetry test. A point generator's fields depend on the sample
    only through u(x, t): wherever two samples have nearly equal u at the
    same (x, t), the fields must nearly agree. Returns the mismatch at such
    point pairs relative to the fields' spread: ~0 pointwise, ~1 unrelated.
    NaN when too few matching pairs exist."""
    c = region(crop)
    u = u_ref[c]
    d = delta_frac * u.std()
    num = den = 0.0
    npair = 0
    for f in gen:
        f = f[c]
        mu = f.mean()
        for r in range(u.shape[0]):
            for s_ in range(r + 1, u.shape[0]):
                m = np.abs(u[r] - u[s_]) < d
                npair += int(m.sum())
                num += float(((f[r] - f[s_])[m] ** 2).sum())
                den += float(((f[r] - mu)[m] ** 2).sum() + ((f[s_] - mu)[m] ** 2).sum())
    if npair < 90:
        return float("nan")
    return num / den if den > 0 else 0.0      # constant fields are pointwise


def symbolic_fit(gen, X, T, u_ref, crop):
    """Per-component least squares on MONO across all reference samples."""
    c = region(crop)
    Xb = np.broadcast_to(X, u_ref.shape)[c]
    Tb = np.broadcast_to(T, u_ref.shape)[c]
    A = monomials(Xb, Tb, u_ref[c]).reshape(-1, len(MONO))
    coefs, r2 = [], []
    for comp in gen:
        y = comp[c].ravel()
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ w
        coefs.append(w)
        r2.append(1 - res @ res / (np.sum((y - y.mean()) ** 2) + 1e-30))
    return coefs, r2


def analytic_split(coefs, u_ref, x, t, crop, M, G, names):
    """Analytic symmetry test of a symbolically fitted generator.

    coefs: fitted xi, eta, phi coefficients on MONO. Coefficients are put on
    a field scale by the RMS of each monomial over the data, then split into
    the kernel of the determining equations (an exact symmetry) and the
    rest. No derivative of any data field is involved.
    """
    X, T = np.meshgrid(x, t, indexing="ij")
    c = region(crop)
    Xb = np.broadcast_to(X, u_ref.shape)[c]
    Tb = np.broadcast_to(T, u_ref.shape)[c]
    rms = np.sqrt(np.mean(monomials(Xb, Tb, u_ref[c]).reshape(-1, len(MONO)) ** 2,
                          axis=0))
    s = np.concatenate([rms, rms, rms])
    cp = np.concatenate(coefs) * s
    Gp = G * s[:, None]
    Qk, _ = np.linalg.qr(Gp)
    ksym = Qk @ (Qk.T @ cp)
    off = np.linalg.norm(cp - ksym) / (np.linalg.norm(cp) + 1e-30)
    w, *_ = np.linalg.lstsq(Gp, ksym, rcond=None)
    return off, w / (np.max(np.abs(w)) + 1e-30)


def algebra_split(gen, gt, crop):
    c = region(crop)
    v = np.concatenate([q[c].ravel() for q in gen])
    B = np.stack([np.concatenate([q[c].ravel() for q in g]) for g in gt]).T
    w, *_ = np.linalg.lstsq(B, v, rcond=None)
    outside = np.linalg.norm(v - B @ w) / (np.linalg.norm(v) + 1e-30)
    return w, outside


def band_random(gt, rng):
    """Smooth random generator with the GT-mean spectrum, per sample."""
    out = []
    for comp in range(3):
        amp = np.sqrt(np.mean([np.abs(np.fft.fft2(g[comp][0])) ** 2
                               for g in gt], axis=0))
        en = np.mean([np.mean(g[comp] ** 2) for g in gt])
        w = np.fft.fft2(rng.standard_normal(amp.shape))
        f = np.real(np.fft.ifft2(w / (np.abs(w) + 1e-30) * amp))
        f = f * np.sqrt(en / (np.mean(f ** 2) + 1e-30))
        out.append(np.broadcast_to(f, gt[0][0].shape).copy())
    return tuple(out)


def fmt_formula(coefs, r2, top=3):
    parts = []
    for name, w, q in zip(("xi", "eta", "phi"), coefs, r2):
        s = np.max(np.abs(w)) + 1e-30
        terms = [f"{w[i] / s:+.2f}*{MONO[i]}"
                 for i in np.argsort(-np.abs(w))[:top] if abs(w[i]) > 0.05 * s]
        parts.append(f"{name}~[{' '.join(terms)}] (R2 {q:.2f})")
    return "   ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--method", default="FINS")
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--crop", type=int, default=8)
    ap.add_argument("--eval-seam", type=float, default=None,
                    help="evaluate only on x in [s L, (1-s) L], the region the "
                         "short-horizon loss constrains (default: crop only)")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    args = ap.parse_args()
    path = args.artifact if os.path.isabs(args.artifact) else os.path.join(
        os.environ.get("FINS_RESULTS_DIR", "."), args.artifact)
    z = np.load(path, allow_pickle=True)
    if str(z["pde"]) != "burgers":
        raise SystemExit("exp29 is written for Burgers (its only non-affine "
                         "symmetry is projective)")
    if "u_ref" not in z.files or f"{args.method}_xi_ref" not in z.files:
        raise SystemExit("artifact lacks the multi-sample block")
    nu = args.nu if args.nu is not None else ExperimentConfig().nu_burgers
    x, t, u_ref = z["x"], z["t"], z["u_ref"]
    n_ref = u_ref.shape[0]
    X, T = np.meshgrid(x, t, indexing="ij")
    names = list(z["gt_names"])
    gt = [tuple(np.stack([np.asarray(GroundTruthAlgebra.burgers_generators(
        x, t, u_ref[r])[k][c]) for r in range(n_ref)]) for c in range(3))
        for k in range(len(names))]
    m = args.method
    fins = [tuple(z[f"{m}_{c}_ref"][k] for c in ("xi", "eta", "phi"))
            for k in range(z[f"{m}_xi_ref"].shape[0])]
    norms = np.asarray(z[f"{m}_norms"]) if f"{m}_norms" in z.files \
        else np.ones(len(fins))

    # the combination inside the method's span that best aligns with the
    # non-affine part of the algebra: the object the span test credited
    Lx = x[-1] + (x[1] - x[0])
    tsl = slice(args.crop, -args.crop)
    if args.eval_seam is not None:
        # score only where the loss constrains the generator
        idx = np.where((x >= args.eval_seam * Lx) & (x <= (1 - args.eval_seam) * Lx))[0]
        xsl = slice(int(idx[0]), int(idx[-1]) + 1)
        print(f"evaluation region: x in [{x[xsl][0]:.2f}, {x[xsl][-1]:.2f}] "
              f"(the loss window, seam {args.eval_seam}), t cropped {args.crop}")
    else:
        xsl = slice(args.crop, -args.crop)
        print(f"evaluation region: x and t cropped {args.crop} per edge")
    ev = (xsl, tsl)
    cr = (slice(None), xsl, tsl)
    Zc = np.zeros_like(X[xsl, tsl])
    aff = []
    for comp in range(3):
        for mono in (np.ones_like(X), X, T):
            parts = [Zc, Zc, Zc]
            parts[comp] = mono[xsl, tsl]
            aff.append(np.concatenate([np.broadcast_to(p, (n_ref,) + p.shape).ravel()
                                       for p in parts]))
        parts = [np.zeros_like(u_ref[cr])] * 3
        parts = list(parts)
        parts[comp] = u_ref[cr]
        aff.append(np.concatenate([p.ravel() for p in parts]))
    Qa, _ = np.linalg.qr(np.stack(aff).T)
    vec = lambda g: np.concatenate([q[cr].ravel() for q in g])
    Bgt = np.stack([vec(g) for g in gt]).T
    Pgt = Bgt - Qa @ (Qa.T @ Bgt)
    Un, sn, _ = np.linalg.svd(Pgt, full_matrices=False)
    ndir = Un[:, 0]
    F = np.stack([vec(g) for g in fins]).T
    Fp = F - Qa @ (Qa.T @ F)
    wbest, *_ = np.linalg.lstsq(Fp, ndir, rcond=None)
    align = float(np.linalg.norm(Fp @ wbest) / (np.linalg.norm(ndir) + 1e-30))
    vstar = tuple(sum(wbest[k] * fins[k][c] for k in range(len(fins)))
                  for c in range(3))

    rng = np.random.default_rng(0)
    Xs = np.broadcast_to(X, u_ref.shape)
    Ts = np.broadcast_to(T, u_ref.shape)
    Z = np.zeros_like(u_ref)
    P = gt[names.index("projective")]
    controls = {f"GT {n}": g for n, g in zip(names, gt)}
    controls["projective, eta removed"] = (P[0], Z, P[2])
    controls["fragment xi = x t only"] = (Xs * Ts, Z, Z)
    for i in range(3):
        controls[f"random smooth #{i + 1}"] = band_random(gt, rng)
    # non-pointwise control: phi reads u an eighth of the domain away
    controls["NONLOCAL: phi = u(x + L/8)"] = (Z, Z, np.roll(u_ref, u_ref.shape[1] // 8, axis=1))

    todo = [(f"{m} v{k + 1} (norm {norms[k]:.2f})", g) for k, g in enumerate(fins)]
    todo.append((f"{m} best non-affine combination (align {align:.2f})", vstar))
    todo += list(controls.items())
    a_coef = 2 * np.pi
    b_coef = a_coef ** 2 * nu
    res_loc = local_residuals([g for _, g in todo], u_ref, x, t, a_coef,
                              b_coef, margin=args.margin)
    res = exact_residuals([g for _, g in todo], u_ref, x, t, nu,
                          margin=args.margin)
    res_sh = shoot_residuals([g for _, g in todo], u_ref, x, t, nu,
                             K=args.K, every=args.every, seam=args.seam)

    print("=" * 100)
    print(f"exp29 -- {os.path.basename(path)}: {m} generators through the EXACT "
          f"solver (nu = {nu}, seam window m = {args.margin} L, {n_ref} samples)")
    print("=" * 100)
    print("local = determining equation Q_t = L_u[Q] on the data (the "
          "criterion for scaling/projective on a torus); flow = the exact "
          "solver's flow map (valid for generators compatible with the "
          "periodic problem only)")
    print(f"shoot = the training loss (K = {args.K}, shooting every "
          f"{args.every}, seam {args.seam}) through the EXACT solver, in the "
          f"units of the run's `equiv`")
    print("nonloc = mismatch of the fields between samples at points where "
          "their u nearly agree: ~0 for a point symmetry, ~1 for a field "
          "shaped per sample")
    print(f"{'generator':<50}{'local':>10}{'flow':>11}{'shoot':>11}{'nonloc':>9}")
    for (lab, g), rl, r, rs in zip(todo, res_loc, res, res_sh):
        nl = nonlocality(g, u_ref, ev)
        print(f"  {lab:<48}{np.median(rl):>10.3e}{np.median(r):>11.3e}{rs:>11.3e}{nl:>9.3f}")
    Mdet = determining_matrix(a_coef, b_coef)
    Gdet = gt_coefficients(a_coef)
    print("\nANALYTIC check (determining equations, exact): the fitted formula is "
          "split into the symmetry algebra and the rest; 'non-symmetry' is the "
          "share of the fitted generator that the determining equations reject")
    rows = [(l, g) for l, g in todo if l.startswith(m) or not l.startswith("GT")]
    for lab, g in rows:
        coefs, r2 = symbolic_fit(g, X, T, u_ref, ev)
        off, wk = analytic_split(coefs, u_ref, x, t, ev, Mdet, Gdet, names)
        w, outside = algebra_split(g, gt, ev)
        print(f"  {lab}")
        print(f"      {fmt_formula(coefs, r2)}")
        print(f"      analytic non-symmetry share {off:.3f}  | symmetry part = "
              + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, wk))
              + f"   | field outside span(GT) {outside:.2f}")


if __name__ == "__main__":
    main()
