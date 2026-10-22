"""Experiment 24 — does the seam mechanism survive a TRAINED surrogate?

exp21/exp22 settled the mechanism at the level of the
condition: on single-scale data a seam-aware local residual admits the
whole heat algebra, a periodic global one does not. exp23 showed the
matching directional effect in a neural discovery loop (scaling, 3/3
seeds) but with an EXACT operator, so surrogate fidelity was never tested
-- and LieSym's own claim is about what a learned operator can see.

This script closes that gap and puts the competing remedies side by side
on one footing. It trains N_theta[u] ~ u_t (the LieSym formulation) and
then runs discovery against it, over the cells:

  base        single scale, no coord channels, spectral residual
  coord       single scale, coord channels, spectral residual
  seam        single scale, coord channels, seam-aware local residual
  ms_coord    MULTI-scale + coord channels, spectral residual  <- LieSym
  ms_seam     multi-scale + coord channels, seam-aware residual

The comparison that matters is seam vs ms_coord: if the single-scale
seam-aware cell matches or beats the multi-scale one, the multi-scale
construction is redundant at its own headline target, and its cost (twice
the GPU budget, the 3500-epoch surrogate, the direction-collapse
instability of section 5.3) is avoidable.

Both metrics are recorded per slot, fields are serialised, and
exp23_stats.py does the aggregation. Nothing here reports a single seed.

Run (GPU):  python exp24_surrogate_protocol.py --cell seam --seeds 42,43,44
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from exp23_single_scale_discovery import (AL, L, T, NX, NT, GenNet, d_local,
                                          d_spectral_x, deflate, gt_generators)

SCALES = [0.5, 0.7, 1.0, 1.4, 2.0]
CELLS = {                       # coords, protocol, multiscale
    "base": (False, "spectral", False),
    "coord": (True, "spectral", False),
    "seam": (True, "local", False),
    "ms_coord": (True, "spectral", True),
    "ms_seam": (True, "local", True),
}


def make_data(n, seed, multiscale):
    """Exact heat trajectories; multiscale varies L on the scaling orbit."""
    rng = np.random.default_rng(seed)
    us, uts, uxs, Xs, Ts = [], [], [], [], []
    for i in range(n):
        lam = SCALES[i % len(SCALES)] if multiscale else 1.0
        Li, Ti = L * lam, T * lam ** 2          # heat orbit: T ~ L^2
        x = np.linspace(0, Li, NX, endpoint=False)
        t = np.linspace(0, Ti, NT)
        X, Tm = np.meshgrid(x, t, indexing="ij")
        u = np.zeros_like(X)
        ux = np.zeros_like(X)
        ut = np.zeros_like(X)
        for k in range(1, 7):
            a = rng.standard_normal() / k ** 1.5
            p = rng.uniform(0, 2 * np.pi)
            kk = 2 * np.pi * k / Li
            dec = np.exp(-AL * kk ** 2 * Tm)
            u += a * dec * np.cos(kk * X + p)
            ux += -a * kk * dec * np.sin(kk * X + p)
            ut += -AL * kk ** 2 * a * dec * np.cos(kk * X + p)
        us.append(u), uts.append(ut), uxs.append(ux), Xs.append(X), Ts.append(Tm)
    f = lambda z: torch.tensor(np.stack(z), dtype=torch.float32)
    return f(us), f(uxs), f(uts), f(Xs), f(Ts)


class Surrogate(nn.Module):
    """N_theta[u] ~ u_t, with optional coordinate channels."""

    def __init__(self, coords, hidden=64, modes=16):
        super().__init__()
        self.coords, self.modes = coords, modes
        cin = 3 if coords else 1
        self.lift = nn.Conv2d(cin, hidden, 1)
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(2, hidden, hidden, modes,
                                     dtype=torch.float32) * 0.02)
            for _ in range(2)])
        self.loc = nn.ModuleList([nn.Conv2d(hidden, hidden, 1)
                                  for _ in range(2)])
        self.proj = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, 1, 1))

    def spectral(self, h, wr):
        fh = torch.fft.rfft(h, dim=2)
        m = min(self.modes, fh.shape[2])
        wc = torch.complex(wr[0, :, :, :m], wr[1, :, :, :m])
        out = torch.zeros_like(fh)
        out[:, :, :m] = torch.einsum("bixy,iox->boxy", fh[:, :, :m], wc)
        return torch.fft.irfft(out, n=h.shape[2], dim=2)

    def forward(self, u, X, Tm):
        ch = [u.unsqueeze(1)]
        if self.coords:
            ch += [(X / X.amax()).unsqueeze(1), (Tm / Tm.amax()).unsqueeze(1)]
        h = self.lift(torch.cat(ch, dim=1))
        for wr, lc in zip(self.w, self.loc):
            h = torch.nn.functional.gelu(self.spectral(h, wr) + lc(h))
        return self.proj(h)[:, 0]


def train_surrogate(S, u, X, Tm, ut, epochs, lr=2e-3, verbose=True):
    opt = torch.optim.AdamW(S.parameters(), lr=lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scale = ut.abs().mean()
    for ep in range(epochs):
        loss = ((S(u, X, Tm) - ut) / scale).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if verbose and (ep + 1) % max(epochs // 5, 1) == 0:
            print(f"    surrogate ep {ep + 1}/{epochs}  "
                  f"rel loss {float(loss):.4f}", flush=True)
    with torch.no_grad():
        rel = float((S(u, X, Tm) - ut).pow(2).mean().sqrt()
                    / (ut.pow(2).mean().sqrt() + 1e-12))
    return rel


def residual_through(S, Q, u, X, Tm, protocol, margin, dx, dt, exact):
    """v[N[u] - u_t] as a directional derivative, per Olver via JVP."""
    Qt = d_local(Q, dt, 2, 1)
    if exact:
        dN = AL * (d_spectral_x(Q, 2) if protocol == "spectral"
                   else d_local(Q, dx, 1, 2))
    else:
        _, dN = torch.func.jvp(lambda z: S(z, X, Tm), (u,), (Q,))
    M = Qt - dN
    i0 = int(margin * NX)
    i1 = NX - i0 if i0 else NX
    return M[:, i0:i1, 2:NT - 2], (slice(None), slice(i0, i1),
                                   slice(2, NT - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="seam", choices=list(CELLS))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--ep-sur", type=int, default=1500)
    ap.add_argument("--n-sym", type=int, default=5)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--exact", action="store_true",
                    help="bypass the surrogate (exact operator) to "
                         "reproduce the exp23 regime for reference")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coords, protocol, multiscale = CELLS[args.cell]
    marg = args.margin if protocol == "local" else 0.0
    outdir = args.outdir or os.environ.get("FINS_RESULTS_DIR", "results")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 88)
    print(f"EXP 24 - cell '{args.cell}': coords={coords} "
          f"protocol={protocol} multiscale={multiscale} "
          f"margin={marg} operator={'exact' if args.exact else 'surrogate'} "
          f"device={dev}")
    print("=" * 88, flush=True)

    for seed in [int(v) for v in args.seeds.split(",")]:
        t_all = time.time()
        u, ux, ut, X, Tm = make_data(args.n_train, seed, multiscale)
        u, ux, ut, X, Tm = [z.to(dev) for z in (u, ux, ut, X, Tm)]
        dx = float(X[0, 1, 0] - X[0, 0, 0])
        dt = float(Tm[0, 0, 1] - Tm[0, 0, 0])
        gt = gt_generators(X[0], Tm[0], u)

        S = Surrogate(coords).to(dev)
        if args.exact:
            delta = 0.0
            print(f"  seed {seed}: exact operator")
        else:
            torch.manual_seed(seed)
            delta = train_surrogate(S, u, X, Tm, ut, args.ep_sur)
            print(f"  seed {seed}: surrogate rel error delta = {delta:.4f}",
                  flush=True)
        for p in S.parameters():
            p.requires_grad_(False)

        frozen, fields = [], []
        for k in range(args.n_sym):
            torch.manual_seed(seed * 100 + k)
            net = GenNet(coords).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3,
                                    weight_decay=1e-5)
            for _ in range(args.iters):
                xi, eta, phi = net(u, X[0], Tm[0])
                v_raw = torch.stack([xi, eta, phi], 1)
                v = deflate(v_raw, frozen)
                Q = v[:, 2] - v[:, 0] * ux - v[:, 1] * ut
                M, sl = residual_through(S, Q, u, X, Tm, protocol, marg,
                                         dx, dt, args.exact)
                den = v[:, :, sl[1], sl[2]].pow(2).mean()
                loss = (M.pow(2).mean() / (den + 1e-9)
                        + (v_raw.pow(2).mean().sqrt() - 1.0).pow(2)
                        + 10.0 * torch.relu(
                            0.1 - v.pow(2).mean().sqrt()).pow(2))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            with torch.no_grad():
                xi, eta, phi = net(u, X[0], Tm[0])
                v = deflate(torch.stack([xi, eta, phi], 1), frozen)
            fields.append(v.detach())
            frozen.append(v[:1].detach())
            a = torch.cat([c.reshape(-1) for c in (v[:, 0], v[:, 1], v[:, 2])])
            a = a / (a.norm() + 1e-12)
            best = max(((abs(float(a @ (torch.cat([c.reshape(-1) for c in g])
                        / (torch.cat([c.reshape(-1) for c in g]).norm()
                           + 1e-12)))), n) for n, g in gt.items()))
            print(f"    slot {k + 1}: -> {best[1]} (cos {best[0]:.3f})",
                  flush=True)

        path = os.path.join(outdir, f"exp24_{args.cell}_s{seed}.npz")
        np.savez(path,
                 fields=torch.stack([torch.stack([f[:, 0], f[:, 1], f[:, 2]])
                                     for f in fields]).cpu().numpy(),
                 u=u.cpu().numpy(),
                 x=X[0, :, 0].cpu().numpy(), t=Tm[0, 0].cpu().numpy(),
                 delta=delta, cell=args.cell, n_sym=args.n_sym)
        print(f"  seed {seed} done in {time.time() - t_all:.0f} s -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
