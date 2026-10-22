"""closure_metrics — a closure statistic that survives the ACE audit.

The published ACE (`Metrics.bracket_closure_error`) fails in three
independent ways, each demonstrated numerically:

1. WRONG BRACKET. The directional derivative omits the phi d/du term and
   differentiates on-shell fields, and the x-difference wraps periodically
   across the seam on non-periodic coefficients. The EXACTLY closed 6-dim
   heat algebra scores 8.24 instead of 0, and the artefact depends on the
   set's content, so it cannot be subtracted as an offset.
2. NOT SCALE-INVARIANT. The bracket is bilinear and the residual is
   absolute, so scaling a set by s scales ACE by s^2: near-zero generators
   score near-perfect closure for free.
3. BLIND TO DEGENERACY. Closure is measured as "how much of the bracket
   escapes the span". If the brackets VANISH there is nothing to escape,
   so an abelian set scores a perfect 0 -- including the superposition
   subalgebra beta(x,t) d_u of a linear PDE, which is precisely the
   documented failure mode, and including k copies of one generator.

Fixes. The bracket is taken in the (xi, eta, a, b) representation with
phi = a(x,t) u + b(x,t), where d(phi)/du = a is known rather than guessed;
that class contains the whole point algebra of each benchmark PDE and is
closed under the bracket (the a1 a2 terms cancel). Generators are
normalised before bracketing, which removes the scale. And closure is
reported as a PAIR, because closure and non-triviality are orthogonal
properties that no single number can carry:

    escaped     mean over pairs of ||[vi,vj] - P_span [vi,vj]|| / ||[vi,vj]||
                -- 0 means closed, 1 means the bracket leaves the span
                entirely; computed only over pairs whose bracket is not
                numerically zero
    structure   mean ||[vi,vj]|| over pairs of unit generators -- how much
                algebra there is to close at all; ~0 flags an abelian or
                duplicated set, which `escaped` alone would call perfect.
                It is NOT a quality score: the true heat algebra sits at
                1.7e-3 and pure noise at 0.7, because random fields have
                far larger brackets than a real algebra. Read it only as
                "is there anything to close", against the reference value
                of the correct algebra on the same grid.

Recovering a and b needs the fields on SEVERAL samples with different u,
since a single sample cannot separate a*u from b. `fit_affine` does that
by per-point least squares and also returns how strongly xi and eta vary
across samples: a genuine point symmetry of this class has xi = xi(x,t),
so a large spread means the discovered object is not one.
"""

import numpy as np


def _d1(f, h, axis):
    """Non-periodic first difference; one-sided at both ends."""
    f = np.moveaxis(f, axis, 0)
    o = np.zeros_like(f)
    o[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    o[0] = (f[1] - f[0]) / h
    o[-1] = (f[-1] - f[-2]) / h
    return np.moveaxis(o, 0, axis)


def bracket_affine(g1, g2, dx, dt):
    """[v1, v2] for v = xi d_x + eta d_t + (a u + b) d_u, exactly."""
    x1, e1, a1, b1 = g1
    x2, e2, a2, b2 = g2
    V1 = lambda f: x1 * _d1(f, dx, 0) + e1 * _d1(f, dt, 1)
    V2 = lambda f: x2 * _d1(f, dx, 0) + e2 * _d1(f, dt, 1)
    return (V1(x2) - V2(x1), V1(e2) - V2(e1), V1(a2) - V2(a1),
            V1(b2) - V2(b1) + b1 * a2 - b2 * a1)


def align_refs(xi, eta, phi):
    """Match generators across reference samples before the affine fit.

    Fields arrive as [n_gens, n_ref, Nx, Nt]. For methods that re-derive
    their generators per sample rather than re-evaluating one trained
    object -- LieNLSD re-solves the determining system -- the ORDER of the
    set is not stable, so regressing phi across samples would mix
    different generators and report nonsense. Each sample's set is matched
    to sample 0's by |cos| (greedy, which is exact enough at these sizes)
    and sign-corrected.
    """
    n_gen, n_ref = xi.shape[0], xi.shape[1]
    out = [a.copy() for a in (xi, eta, phi)]
    flat = lambda r: np.stack([
        np.concatenate([xi[g, r].ravel(), eta[g, r].ravel(),
                        phi[g, r].ravel()]) for g in range(n_gen)])
    A0 = flat(0)
    A0 = A0 / (np.linalg.norm(A0, axis=1, keepdims=True) + 1e-30)
    for r in range(1, n_ref):
        Ar = flat(r)
        Ar = Ar / (np.linalg.norm(Ar, axis=1, keepdims=True) + 1e-30)
        sim = A0 @ Ar.T
        taken, perm, sign = set(), [0] * n_gen, [1.0] * n_gen
        for g in np.argsort(-np.abs(sim).max(axis=1)):
            order = np.argsort(-np.abs(sim[g]))
            pick = next((j for j in order if j not in taken), order[0])
            taken.add(pick)
            perm[g] = pick
            sign[g] = 1.0 if sim[g, pick] >= 0 else -1.0
        for a, src in zip(out, (xi, eta, phi)):
            a[:, r] = np.stack([sign[g] * src[perm[g], r]
                                for g in range(n_gen)])
    return out


def fit_affine(xi, eta, phi, u):
    """Recover (xi, eta, a, b) from per-sample fields.

    xi, eta, phi, u all have shape [n_samples, Nx, Nt]. phi is regressed
    on u pointwise: phi_i = a * u_i + b. Returns the generator plus the
    relative spread of xi and eta across samples, which should be ~0 for a
    point symmetry of this class.
    """
    n = u.shape[0]
    if n < 2:
        raise ValueError(
            "the affine fit needs fields on at least 2 samples with "
            "different u; a single sample cannot separate a*u from b")
    um = u.mean(0)
    pm = phi.mean(0)
    cov = ((u - um) * (phi - pm)).mean(0)
    var = ((u - um) ** 2).mean(0)
    a = np.where(var > 1e-12 * var.max(), cov / (var + 1e-30), 0.0)
    b = pm - a * um
    spread = float(np.mean(xi.std(0)) / (np.abs(xi).mean() + 1e-12)
                   + np.mean(eta.std(0)) / (np.abs(eta).mean() + 1e-12)) / 2
    return (xi.mean(0), eta.mean(0), a, b), spread


def project_polynomial(gen_ab, x, t, deg=2):
    """Least-squares projection of each coefficient onto monomials x^p t^q.

    The bracket differentiates, and differentiation amplifies field noise
    by ~1/dx, so on raw discovered fields the closure statistic saturates:
    1 percent noise already sends `escaped` to 0.8. Projecting the
    coefficients onto a low-order polynomial basis first asks the question
    that actually matters -- is the STRUCTURE closed -- instead of
    measuring the discretisation noise. The projection is exact for every
    generator of the heat, Burgers and KdV point algebras, all of whose
    coefficients are polynomials of degree <= 2 in (x, t).
    """
    X, Tm = np.meshgrid(x, t, indexing="ij")
    B = np.stack([(X ** p * Tm ** q).ravel()
                  for p in range(deg + 1) for q in range(deg + 1 - p)], 1)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
    out = []
    for c in gen_ab:
        co, *_ = np.linalg.lstsq(Bn, c.ravel(), rcond=None)
        out.append((Bn @ co).reshape(c.shape))
    return tuple(out)


def closure_report(gens_ab, dx, dt, tol=1e-6, basis=None, deg=2):
    """escaped / structure for a set of (xi, eta, a, b) generators."""
    if basis is not None:                     # basis = (x, t) grids
        gens_ab = [project_polynomial(g, basis[0], basis[1], deg)
                   for g in gens_ab]
    flat = lambda g: np.concatenate([c.ravel() for c in g])
    V = np.stack([flat(g) for g in gens_ab])
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / (nrm + 1e-30)
    gens = [tuple(c / (float(nrm[i, 0]) + 1e-30) for c in g)
            for i, g in enumerate(gens_ab)]
    G = V.T
    esc, mag = [], []
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            b = flat(bracket_affine(gens[i], gens[j], dx, dt))
            nb = float(np.linalg.norm(b))
            mag.append(nb)
            if nb <= tol:
                continue
            c, *_ = np.linalg.lstsq(G, b, rcond=None)
            esc.append(float(np.linalg.norm(b - G @ c) / nb))
    return {
        "escaped": float(np.mean(esc)) if esc else float("nan"),
        "structure": float(np.mean(mag)),
        "abelian_pairs": int(sum(1 for m in mag if m <= tol)),
        "n_pairs": len(mag),
    }


def describe(rep):
    """One-line verdict, with the degenerate cases named explicitly."""
    if rep["abelian_pairs"] == rep["n_pairs"]:
        return ("DEGENERATE: every bracket vanishes (abelian or duplicated "
                "set); closure is trivially satisfied and means nothing")
    if rep["structure"] < 1e-4:
        return ("NEAR-DEGENERATE: brackets are numerically negligible; "
                "read `escaped` with suspicion")
    if np.isnan(rep["escaped"]):
        return "no usable pairs"
    q = rep["escaped"]
    kind = ("closed" if q < 0.05 else
            "mostly closed" if q < 0.25 else
            "partially closed" if q < 0.6 else "not closed")
    return (f"{kind}: {100 * q:.0f}% of the average bracket leaves the "
            f"span, structure {rep['structure']:.2f}")
