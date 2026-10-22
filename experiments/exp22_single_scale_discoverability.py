"""Experiment 22 — is the heat algebra discoverable at SINGLE scale?

LieSym §5.2 claims single-domain-length heat data is physically degenerate
for the scaling generator x d_x + 2t d_t, and that "no operator-learning
method can distinguish heat at (alpha, L=1) from heat at (alpha, L=sqrt2)
from the u-fields alone" -- hence multi-scale + coord-aware data is
REQUIRED. exp21 tested the condition on the ground-truth generators; this
script tests DISCOVERY, in the strongest form available.

Method. Parameterise the generator linearly,

    xi  = sum_i a_i m_i(x,t)
    eta = sum_i b_i m_i(x,t)
    phi = sum_i c_i m_i(x,t) + sum_i d_i m_i(x,t) u

over monomials m_i = x^p t^q with p+q <= 2. This ansatz contains the whole
6-dimensional heat algebra exactly. Because the characteristic
Q = phi - xi u_x - eta u_t and the Olver operator M[Q] = D_t Q - alpha D_xx Q
are both LINEAR in the coefficients, the set of discoverable generators is
exactly the null space of the design matrix -- obtainable by SVD, with no
optimiser, no architecture and no training noise to confound the answer.
A method that cannot find what sits in this null space is limited by its
own optimisation, not by the data.

The coefficient space is whitened by the Gram matrix of the basis fields,
so singular values are comparable to the L2 norm of the generator itself
and "unit norm" means the same thing it means everywhere else in the
project.

The ablation is the evaluation protocol only -- the DATA is identical and
single-scale throughout:

  spectral        periodic FFT derivatives (what a translation-equivariant
                  global operator applies); x-weighted characteristics are
                  seam-discontinuous, so the seam pollutes every point
  local, m        non-periodic finite differences, rows restricted to
                  x outside a margin m*L around the seam

Run:  python exp22_single_scale_discoverability.py
"""

import numpy as np

AL, L, T, NX, NT, NSAMP, KMAX = 0.01, 2 * np.pi, 1.0, 64, 128, 6, 6
DEG = 2


def heat_sample(rng, x, t):
    """Exact periodic heat solution and its analytic u_x, u_t."""
    X, Tm = np.meshgrid(x, t, indexing="ij")
    u = np.zeros_like(X)
    ux = np.zeros_like(X)
    ut = np.zeros_like(X)
    for k in range(1, KMAX + 1):
        a = rng.standard_normal() / k ** 1.5
        p = rng.uniform(0, 2 * np.pi)
        dec = np.exp(-AL * k ** 2 * Tm)
        u += a * dec * np.cos(k * X + p)
        ux += -a * k * dec * np.sin(k * X + p)
        ut += -AL * k ** 2 * a * dec * np.cos(k * X + p)
    return u, ux, ut


def monomials(x, t):
    X, Tm = np.meshgrid(x, t, indexing="ij")
    out, names = [], []
    for p in range(DEG + 1):
        for q in range(DEG + 1 - p):
            out.append(X ** p * Tm ** q)
            names.append(f"x^{p} t^{q}")
    return out, names


def d_local(f, h, axis, order):
    f = np.moveaxis(f, axis, 0)
    o = np.zeros_like(f)
    if order == 1:
        o[1:-1] = (f[2:] - f[:-2]) / (2 * h)
        o[0] = (f[1] - f[0]) / h
        o[-1] = (f[-1] - f[-2]) / h
    else:
        o[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2
        o[0], o[-1] = o[1], o[-2]
    return np.moveaxis(o, 0, axis)


def d_spectral(f, axis, order, span):
    n = f.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=span / n)
    sh = [1, 1]
    sh[axis] = n
    return np.real(np.fft.ifft((1j * k.reshape(sh)) ** order
                               * np.fft.fft(f, axis=axis), axis=axis))


def basis_fields(mons, u, ux, ut, free_b=True):
    """(xi, eta, phi) and the resulting Q for every coefficient.

    free_b=False drops the phi = m(x,t) block, i.e. the b term of the
    constrained ansatz phi = a(x,t) u + b(x,t). That block is the
    superposition subalgebra of a LINEAR PDE -- exact spurious symmetries
    (exp2) -- so dropping it is what the b-penalty of exp6 achieves, and
    it is what makes the recovered dimension comparable to the 6 of the
    point algebra.
    """
    Z = np.zeros_like(u)
    trip, Q = [], []
    for m in mons:                                   # xi = m
        trip.append((m, Z, Z))
        Q.append(-m * ux)
    for m in mons:                                   # eta = m
        trip.append((Z, m, Z))
        Q.append(-m * ut)
    if free_b:
        for m in mons:                               # phi = m
            trip.append((Z, Z, m))
            Q.append(m.copy())
    for m in mons:                                   # phi = m * u
        trip.append((Z, Z, m * u))
        Q.append(m * u)
    return trip, Q


def olver(Q, x, t, mode):
    dx, dt = x[1] - x[0], t[1] - t[0]
    Qt = d_local(Q, dt, 1, 1)
    Qxx = (d_spectral(Q, 0, 2, L) if mode == "spectral"
           else d_local(Q, dx, 0, 2))
    return Qt - AL * Qxx


def gt_coeffs(names, n, free_b=True):
    """Ground truth in the coefficient basis (blocks: xi, eta, [phi,] phi*u)."""
    nb = len(names)
    ublk = 3 * nb if free_b else 2 * nb
    idx = {nm: i for i, nm in enumerate(names)}
    g = {k: np.zeros(n) for k in
         ("d_x", "d_t", "u d_u", "Galilean", "scaling", "projective")}
    g["d_x"][idx["x^0 t^0"]] = 1.0
    g["d_t"][nb + idx["x^0 t^0"]] = 1.0
    g["u d_u"][ublk + idx["x^0 t^0"]] = 1.0
    g["Galilean"][idx["x^0 t^1"]] = 2 * AL
    g["Galilean"][ublk + idx["x^1 t^0"]] = -1.0
    g["scaling"][idx["x^1 t^0"]] = 1.0
    g["scaling"][nb + idx["x^0 t^1"]] = 2.0
    g["projective"][idx["x^1 t^1"]] = 4 * AL
    g["projective"][nb + idx["x^0 t^2"]] = 4 * AL
    g["projective"][ublk + idx["x^2 t^0"]] = -1.0
    g["projective"][ublk + idx["x^0 t^1"]] = -2 * AL
    return g


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    mons, names = monomials(x, t)
    nb = len(mons)
    samples = [heat_sample(rng, x, t) for _ in range(NSAMP)]
    protocols = [("spectral (periodic, global)", "spectral", 0.0),
                 ("local FD, m = 0", "local", 0.0),
                 ("local FD, m = 0.05L", "local", 0.05),
                 ("local FD, m = 0.10L", "local", 0.10)]

    print("=" * 94)
    print(f"EXP 22 - exact discoverability of the heat algebra, SINGLE "
          f"scale L = 2pi, Nx = {NX}, {NSAMP} samples, alpha = {AL}")
    print(f"linear ansatz: monomials x^p t^q, p+q <= {DEG}; contains the "
          f"full 6-dim algebra exactly")
    print("=" * 94)

    for free_b in (False, True):
        npar = (4 if free_b else 3) * nb
        print(f"\n{'#' * 94}")
        print(f"# ANSATZ: phi = a(x,t) u"
              + (" + b(x,t)   [free b: includes the superposition "
                 "subalgebra]" if free_b else
                 "            [constrained, b = 0: point algebra only]")
              + f"   ({npar} coefficients)")
        print("#" * 94)
        run_protocols(protocols, mons, names, samples, x, t, npar, free_b)


def run_protocols(protocols, mons, names, samples, x, t, npar, free_b):
    trip0, _ = basis_fields(mons, *samples[0], free_b=free_b)
    F = np.stack([np.concatenate([c.ravel() for c in tr]) for tr in trip0])
    G = F @ F.T / F.shape[1]
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 1e-14, None)
    Rinv = V @ np.diag(w ** -0.5) @ V.T
    gts = gt_coeffs(names, npar, free_b=free_b)
    tol = 1e-2                                    # the papers' threshold
    for label, mode, marg in protocols:
        rows = []
        for (u, ux, ut) in samples:
            _, Qb = basis_fields(mons, u, ux, ut, free_b=free_b)
            i0, i1 = int(marg * NX), NX - int(marg * NX)
            sl = (slice(i0, i1), slice(2, NT - 2))
            rows.append(np.stack([olver(q, x, t, mode)[sl].ravel()
                                  for q in Qb], axis=1))
        A = np.concatenate(rows, axis=0) @ Rinv
        A /= np.sqrt(A.shape[0])
        _, s_, Vt = np.linalg.svd(A, full_matrices=False)
        k = int((s_ < tol).sum())
        null = Vt[len(s_) - k:] if k else np.zeros((0, npar))
        print(f"\n  {label}")
        print("    smallest singular values: "
              + "  ".join(f"{v:.2e}" for v in s_[-7:]))
        print(f"    admissible dimension at threshold {tol:g}: {k}"
              f"   (point algebra = 6)")
        cont = []
        for nm, g in gts.items():
            gw = np.linalg.solve(Rinv, g)
            gw = gw / (np.linalg.norm(gw) + 1e-30)
            cont.append(f"{nm} {float(np.linalg.norm(null @ gw)) if k else 0:.3f}")
        print("    containment of each GT generator: " + ", ".join(cont))
        print(f"    (random-span reference: {np.sqrt(k / npar):.2f})")


if __name__ == "__main__":
    main()
