"""Time-scaling exponent read from a discovered span: the direction of the plane {x d_x + a t d_t + (1 - a) u d_u}
(spanned by A = x d_x + u d_u and B = t d_t - u d_u, each reduced to its part orthogonal to the translations and the
Galilean boost) that lies deepest inside the span; a-hat = (B coefficient) / (A coefficient). Scaling and metric as
exp51's scorer (band [0.2, 0.8] L, times t_8 .. t_{Nt-8}, components xi / L, eta / T, phi / sigma_u).

Benjamin-Ono a = 2, KdV a = 3, viscous Burgers a = 2, fractional Burgers u_t + u u_x = -nu Lambda^alpha u: a = alpha.

Run:  python exponent_readout.py <npz> [<npz> ...]   (FINS / exp50 / exp56 files)
"""

import sys

import numpy as np

import exp31_polynomial_discovery as e31
import exp42_tables as e42


def plane_and_rest(pde, u, xs, ts, keep, w, cfg):
    """A, B (reduced) and the reference fields that the reduction removes (translations, boost)."""
    names, gt = e42.ground_truth(pde, u, xs, ts, cfg)
    rest = [g for g, nm in zip(gt, names) if nm in ("∂_x", "∂_t", "Galilean", "shift")]
    R = e42.vectors(np.stack(rest), keep, w, 8)
    X, T = np.broadcast_to(xs[:, :, None], u.shape), np.broadcast_to(ts[:, None, :], u.shape)
    A = e42.vectors(np.stack([X, 0 * T, u])[None], keep, w, 8)[0]
    B = e42.vectors(np.stack([0 * X, T, -u])[None], keep, w, 8)[0]
    Qr = e42.orth(R)
    return A - Qr @ (Qr.T @ A), B - Qr @ (Qr.T @ B), Qr


def read(D, A, B, Qr):
    """(a-hat, containment of the best plane direction in span(D) + rest)."""
    Q = e42.orth(np.concatenate([D, Qr.T]) if Qr.size else D)
    P = np.stack([A, B], 1)
    Pn = np.linalg.norm(P, axis=0)
    Po = P / Pn
    M = Po.T @ Q @ (Q.T @ Po)
    G = Po.T @ Po
    # generalised eigenproblem M c = lam G c on the 2-plane (Po is not orthogonal)
    lam, vec = np.linalg.eig(np.linalg.solve(G, M))
    i = int(np.argmax(lam.real))
    c = vec[:, i].real / Pn
    return float(c[1] / c[0]), float(lam[i].real)


def geometry(pde, u, xs, ts, cfg):
    """exp51's scored region and weights, and the reduced plane: (keep, w, A, B)."""
    Nx = u.shape[1]
    keep = (np.arange(Nx) / Nx >= 0.2) & (np.arange(Nx) / Nx <= 0.8)
    w = np.array([1 / ((xs[0, 1] - xs[0, 0]) * Nx), 1 / ts[:, -1].max(), 1 / u.std()])
    A, B, _ = plane_and_rest(pde, u, xs, ts, keep, w, cfg)
    return keep, w, A, B


def read_fields(f, geo):
    """(a-hat, containment) of discovered fields f [k, 3, N, Nx, Nt]."""
    keep, w, A, B = geo
    D = e42.vectors(f, keep, w, 8)
    D = np.where(np.isfinite(D), D, 0.0)
    return read(D, A, B, np.zeros((len(A), 0)))


def main():
    cfg = e31.ExperimentConfig()
    for f in sys.argv[1:]:
        z = np.load(f, allow_pickle=True)
        geo = geometry(str(z["pde"]), z["u"], z["xs"], z["ts"], cfg)
        a, c = read_fields(e42.fins_fields(z), geo)
        print(f"{f}: a-hat {a:.3f} (containment {c:.4f})")


if __name__ == "__main__":
    main()
