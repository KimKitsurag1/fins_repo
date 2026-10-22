"""Experiment 23 — neural discovery at SINGLE scale: coords x seam, 2x2.

exp22 established ADMISSIBILITY: with a seam margin the whole 6-dim heat
algebra lies exactly in the null space of the Olver operator on
single-scale data. exp19 taught that admissibility is not recovery, so
this script runs the actual discovery loop -- a generator network trained
against the Olver residual -- and ablates the two factors LieSym bundles
together in MS+coord, WITHOUT ever varying the domain length:

  coords    whether the generator network sees (x, t) channels at all.
            xi = x is unrepresentable without them, so this is the
            necessary condition LieSym's coord-aware input supplies.
  protocol  spectral = periodic FFT derivatives in the residual, which is
            what a translation-equivariant global operator applies and
            what pollutes every point through the seam;
            local = non-periodic finite differences with the residual
            restricted to x outside a margin around the seam.

If (coords, local) recovers the scaling generator at single scale while
(coords, spectral) does not, then the seam -- not orbit coverage -- is
what MS+coord was fixing, and LieSym's Figure 4 decomposition (multi-scale
alone +0.04, coord channel +0.23) is explained.

The operator is EXACT (no surrogate), so surrogate fidelity cannot
confound the answer; the ansatz is constrained to phi = a(x,t) u, which
excludes the superposition subalgebra (exp2, exp6); the norm anchor sits
on the RAW generator so the deflated norm stays free to collapse (exp15).

Run:  python exp23_single_scale_discovery.py --coords on --protocol local
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

AL, L, T, NX, NT = 0.01, 2 * np.pi, 1.0, 64, 128
KMAX = 6


def make_data(n, seed):
    """Exact periodic heat solutions with analytic u_x, u_t."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    u, ux, ut = [np.zeros((n, NX, NT)) for _ in range(3)]
    for i in range(n):
        for k in range(1, KMAX + 1):
            a = rng.standard_normal() / k ** 1.5
            p = rng.uniform(0, 2 * np.pi)
            dec = np.exp(-AL * k ** 2 * Tm)
            u[i] += a * dec * np.cos(k * X + p)
            ux[i] += -a * k * dec * np.sin(k * X + p)
            ut[i] += -AL * k ** 2 * a * dec * np.cos(k * X + p)
    f = lambda z: torch.tensor(z, dtype=torch.float32)
    return f(u), f(ux), f(ut), f(X), f(Tm), x, t


def gt_generators(X, Tm, u):
    """Per-SAMPLE ground truth: each component has u's shape [n, Nx, Nt].

    Every generator is compared on the whole evaluation set at once (a
    single reference sample adds cross-seed spread).
    """
    Xb, Tb = X.expand_as(u), Tm.expand_as(u)
    Z, O = torch.zeros_like(u), torch.ones_like(u)
    return {
        "d_x": (O, Z, Z),
        "d_t": (Z, O, Z),
        "u d_u": (Z, Z, u),
        "Galilean": (2 * AL * Tb, Z, -Xb * u),
        "scaling": (Xb, 2 * Tb, Z),
        "projective": (4 * AL * Tb * Xb, 4 * AL * Tb ** 2,
                       -(Xb ** 2 + 2 * AL * Tb) * u),
    }


def d_local(f, h, dim, order):
    """Non-periodic differences; one-sided at both ends."""
    f = f.transpose(dim, -1)
    o = torch.zeros_like(f)
    if order == 1:
        o[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * h)
        o[..., 0] = (f[..., 1] - f[..., 0]) / h
        o[..., -1] = (f[..., -1] - f[..., -2]) / h
    else:
        o[..., 1:-1] = (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / h ** 2
        o[..., 0], o[..., -1] = o[..., 1], o[..., -2]
    return o.transpose(dim, -1)


def d_spectral_x(f, order):
    k = 2 * np.pi * torch.fft.fftfreq(NX, d=L / NX)
    k = k.view(1, NX, 1).to(f.dtype)
    return torch.real(torch.fft.ifft((1j * k) ** order
                                     * torch.fft.fft(f + 0j, dim=1), dim=1))


class GenNet(nn.Module):
    """(u[, x, t]) -> (xi, eta, a), with phi = a * u (constrained ansatz)."""

    def __init__(self, coords, hidden=48):
        super().__init__()
        self.coords = coords
        cin = 3 if coords else 1
        pad = "replicate"                       # non-periodic by design
        self.net = nn.Sequential(
            nn.Conv2d(cin, hidden, 3, padding=1, padding_mode=pad), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode=pad),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode=pad),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1))

    def forward(self, u, X, Tm):
        ch = [u.unsqueeze(1)]
        if self.coords:
            b = u.shape[0]
            ch += [X.expand(b, -1, -1).unsqueeze(1) / L,
                   Tm.expand(b, -1, -1).unsqueeze(1) / T]
        out = self.net(torch.cat(ch, dim=1))
        return out[:, 0], out[:, 1], out[:, 2] * u          # xi, eta, phi


def deflate(v, frozen):
    for f in frozen:
        c = (v * f).sum(dim=(1, 2, 3), keepdim=True) / \
            (f.pow(2).sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
        v = v - c * f
    return v


def residual(Q, protocol, margin, dx, dt):
    Qt = d_local(Q, dt, 2, 1)
    Qxx = (d_spectral_x(Q, 2) if protocol == "spectral"
           else d_local(Q, dx, 1, 2))
    M = Qt - AL * Qxx
    i0 = int(margin * NX)
    i1 = NX - i0 if i0 else NX
    return M[:, i0:i1, 2:NT - 2], (slice(None), slice(i0, i1),
                                   slice(2, NT - 2))


def flat(v):
    return v.reshape(v.shape[0], -1)


def cos_to_gt(trip, gt):
    a = torch.cat([c.reshape(-1) for c in trip])
    a = a / (a.norm() + 1e-12)
    best, nm = 0.0, "-"
    for k, g in gt.items():
        b = torch.cat([c.reshape(-1) for c in g])
        b = b / (b.norm() + 1e-12)
        c = abs(float(a @ b))
        if c > best:
            best, nm = c, k
    return nm, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="on", choices=["on", "off"])
    ap.add_argument("--protocol", default="local",
                    choices=["local", "spectral"])
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--n-sym", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="comma list; runs each and prints one line per seed")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.seeds:
        for sd in [int(v) for v in args.seeds.split(",")]:
            args.seed, args.seeds = sd, None
            run_one(args)
        return
    run_one(args)


def run_one(args):

    u, ux, ut, X, Tm, x_np, t_np = make_data(args.n_train, args.seed)
    dx, dt = float(x_np[1] - x_np[0]), float(t_np[1] - t_np[0])
    gt = gt_generators(X, Tm, u)
    marg = args.margin if args.protocol == "local" else 0.0

    if not args.quiet:
        print("=" * 84)
        print(f"EXP 23 - single-scale neural discovery | coords = "
              f"{args.coords} | protocol = {args.protocol} | margin = "
              f"{marg} | threads = {torch.get_num_threads()}")
        print("=" * 84)

    frozen, found, found_fields = [], [], []
    for k in range(args.n_sym):
        t0 = time.time()
        torch.manual_seed(args.seed * 100 + k)
        net = GenNet(args.coords == "on")
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        for it in range(args.iters):
            xi, eta, phi = net(u, X, Tm)
            v_raw = torch.stack([xi, eta, phi], 1)
            v = deflate(v_raw, frozen)
            Q = v[:, 2] - v[:, 0] * ux - v[:, 1] * ut
            M, sl = residual(Q, args.protocol, marg, dx, dt)
            den = v[:, :, sl[1], sl[2]].pow(2).mean()
            l_eq = M.pow(2).mean() / (den + 1e-9)
            l_anchor = (v_raw.pow(2).mean().sqrt() - 1.0).pow(2)
            perp = v.pow(2).mean().sqrt()
            l_floor = torch.relu(0.1 - perp).pow(2)
            (l_eq + l_anchor + 10.0 * l_floor).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            xi, eta, phi = net(u, X, Tm)
            v = deflate(torch.stack([xi, eta, phi], 1), frozen)
            Q = v[:, 2] - v[:, 0] * ux - v[:, 1] * ut
            M, sl = residual(Q, args.protocol, marg, dx, dt)
            r = float(M.pow(2).mean()
                      / (v[:, :, sl[1], sl[2]].pow(2).mean() + 1e-9))
            nrm = float(v.pow(2).mean().sqrt())
        trip = (v[:, 0], v[:, 1], v[:, 2])
        nm, c = cos_to_gt(trip, gt)
        found.append((nm, c))
        found_fields.append(trip)
        frozen.append(v[:1].detach())
        if not args.quiet:
            print(f"  slot {k + 1}: |v_perp| = {nrm:.3f}  R = {r:.2e}  "
                  f"-> {nm} (cos {c:.3f})  [{time.time() - t0:.0f} s]",
                  flush=True)

    hit = {n for n, c in found if c >= 0.7}

    # Per-slot best-match hides what the SET spans (exp19): report the
    # containment of every GT generator in the span of the slots, and the
    # novel component relative to the essential trio, whose correlation
    # with the x-weighted generators inflates the raw number for free.
    A = torch.stack([torch.cat([c.reshape(-1) for c in tr])
                     for tr in found_fields])
    A = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    G = torch.stack([torch.cat([c.reshape(-1) for c in g])
                     for g in gt.values()])
    G = G / (G.norm(dim=1, keepdim=True) + 1e-12)
    _, sv, Vt = torch.linalg.svd(A, full_matrices=False)
    V = Vt[sv / sv[0] > 1e-6]
    names = list(gt.keys())
    ess = [names.index(n) for n in ("d_x", "d_t", "u d_u")]
    Qe, _ = torch.linalg.qr(G[ess].T)
    nov = {}
    for j, n in enumerate(names):
        gn = G[j] - Qe @ (Qe.T @ G[j])
        nf = float(gn.norm())
        nov[n] = float((V @ (gn / nf)).norm()) if nf > 1e-6 else float("nan")
    np.savez(f"results/exp23_{args.coords}_{args.protocol}_s{args.seed}.npz",
             fields=torch.stack([torch.stack(tr) for tr in found_fields])
             .detach().numpy(),
             u=u.numpy(), x=x_np, t=t_np, n_sym=args.n_sym)
    print(f"  [{args.coords:>3} / {args.protocol:<8} seed {args.seed}] "
          f"novel:  Galilean {nov['Galilean']:.3f}   "
          f"scaling {nov['scaling']:.3f}   "
          f"projective {nov['projective']:.3f}   |  hits "
          f"{sorted(hit) if hit else 'none'}", flush=True)


if __name__ == "__main__":
    main()
