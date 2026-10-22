"""Window stepper: a local operator that needs no boundary condition.

Valid (unpadded) 1-D convolutions map u(t) on a window of width W to
u(t + dt) on the central W - 2r points, r the receptive radius. After K
steps the output lives on W - 2rK points: the influence cone is built into
the architecture, and the operator never closes the domain. Trained on
windows cut from trajectories; periodic data may be read as solutions on
the line, windows spanning several periods.

GlobalStepper: same-size, non-periodic global operators on the same windows
(CNO; FNO with zero domain padding and no positional embedding). r = 0: the
output keeps the window's width, and only an interior region is scored.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

from cno1d import CNO1d
from stepper_surrogate import load_partial, save_partial


class WindowStepper(nn.Module):
    def __init__(self, width=64, layers=3, kernel=5):
        super().__init__()
        self.k = kernel // 2
        self.r = layers * self.k
        self.lift = nn.Conv1d(1, width, 1)
        self.body = nn.ModuleList([nn.Conv1d(width, width, kernel) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, u):
        h = self.lift(u)
        for conv in self.body:
            h = F.gelu(conv(h)) + h[:, :, self.k:-self.k]
        return u[:, :, self.r:-self.r] + self.proj(h)


def windows(sols, idx, j0, pos, width, steps):
    """Windows [B, width] at time j0 and targets for 1..steps, wrapping periodic data."""
    Nx = sols.shape[2]
    cols = (pos[:, None] + torch.arange(width, device=sols.device)[None]) % Nx
    rows = idx[:, None].expand_as(cols)
    return [sols[rows, 0, cols, (j0 + k)[:, None].expand_as(cols)] for k in range(steps + 1)]


def train_window_stepper(train_sol, device, epochs=400, unroll=4, out=32, lr=2e-3, batch=256,
                         seed=0, val_frac=0.1, eval_every=10, verbose=True, **arch):
    """Unrolled MSE on windows; returns the state with the best held-out increment error."""
    torch.manual_seed(seed)
    sols = train_sol.to(device)
    n_all, _, Nx, Nt = sols.shape
    n_val = max(1, int(round(val_frac * n_all)))
    stp = WindowStepper(**arch).to(device)
    width = out + 2 * stp.r * unroll
    opt = torch.optim.Adam(stp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_tr = n_all - n_val
    best, best_state = float("inf"), None

    def rollout(idx, j0, pos):
        ws = windows(sols, idx, j0, pos, width, unroll)
        s = ws[0][:, None]
        errs, incs = [], []
        for k in range(1, unroll + 1):
            s = stp(s)
            c = k * stp.r
            tgt = ws[k][:, c:width - c]
            errs.append((s[:, 0] - tgt).pow(2).mean(1))
            incs.append((tgt - ws[0][:, c:width - c]).pow(2).mean(1))
        return torch.stack(errs, 1), torch.stack(incs, 1)

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    vidx = torch.arange(n_tr, n_all).repeat_interleave(16).to(device)
    vj = torch.randint(0, Nt - unroll, (vidx.numel(),), generator=g).to(device)
    vp = torch.randint(0, Nx, (vidx.numel(),), generator=g).to(device)

    def validate():
        stp.eval()
        with torch.no_grad():
            e, c = rollout(vidx, vj, vp)
            v = float((e / (c + 0.1 * c.mean(0, keepdim=True) + 1e-12)).mean().sqrt())
        stp.train()
        return v

    per_epoch = max(n_tr * 8 // batch, 1)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(per_epoch):
            idx = torch.randint(0, n_tr, (batch,), device=device)
            j0 = torch.randint(0, Nt - unroll, (batch,), device=device)
            pos = torch.randint(0, Nx, (batch,), device=device)
            e, _ = rollout(idx, j0, pos)
            loss = e.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            v = validate()
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in stp.state_dict().items()}
        if verbose and (ep == 0 or (ep + 1) % max(epochs // 10, 1) == 0):
            print(f"      window stepper epoch {ep + 1}/{epochs}: train loss {tot / per_epoch:.3e} | "
                  f"held-out increment error {validate():.3f} (best {best:.3f})", flush=True)
    if best_state is not None:
        stp.load_state_dict(best_state)
        if verbose:
            print(f"      window stepper: returning the best held-out state (increment error {best:.3f})",
                  flush=True)
    return stp


class GlobalStepper(nn.Module):
    """u -> u + net(u) on a window of fixed width; net is 'cno' or 'fnopad'."""

    def __init__(self, kind="cno", size=256, cno_layers=3, cno_res=2, cno_mult=16,
                 fno_modes=32, fno_width=48, fno_layers=4, fno_pad=0.25):
        super().__init__()
        self.r = 0
        self.kind = kind
        if kind == "cno":
            self.net = CNO1d(1, 1, size, cno_layers, N_res=cno_res, N_res_neck=cno_res,
                             channel_multiplier=cno_mult, use_bn=False)
        elif kind == "fnopad":
            self.net = FNO(n_modes=(fno_modes,), hidden_channels=fno_width, in_channels=1, out_channels=1,
                           n_layers=fno_layers, domain_padding=fno_pad, positional_embedding=None)
        else:
            raise ValueError(kind)

    def forward(self, u):
        return u + self.net(u)


def to_float64(module):
    """Module.double() leaves complex parameters (spectral weights) in single precision."""
    module.double()
    for p in module.parameters():
        if p.is_complex():
            p.data = p.data.to(torch.complex128)
    for name, b in module.named_buffers():
        if b.is_complex():
            b.data = b.data.to(torch.complex128)
    return module


def train_global_stepper(train_sol, device, kind="cno", size=256, epochs=400, unroll=4, margin=6,
                         lr=1e-3, batch=256, seed=0, val_frac=0.1, eval_every=10, verbose=True, ckpt=None,
                         ckpt_every=10, **arch):
    """Unrolled MSE on windows of `size` points; step k is scored on [margin k, size - margin k],
    where information from outside the window cannot yet have arrived. Returns the state with
    the best held-out increment error on the same interior. ckpt: partial state written every
    ckpt_every epochs and resumed from if present (same result as an uninterrupted run)."""
    torch.manual_seed(seed)
    sols = train_sol.to(device)
    n_all, _, Nx, Nt = sols.shape
    n_val = max(1, int(round(val_frac * n_all)))
    n_tr = n_all - n_val
    stp = GlobalStepper(kind, size, **arch).to(device)
    opt = torch.optim.Adam(stp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None

    def rollout(idx, j0, pos):
        ws = windows(sols, idx, j0, pos, size, unroll)
        s = ws[0][:, None]
        errs, incs = [], []
        for k in range(1, unroll + 1):
            s = stp(s)
            c = k * margin
            tgt = ws[k][:, c:size - c]
            errs.append((s[:, 0, c:size - c] - tgt).pow(2).mean(1))
            incs.append((tgt - ws[0][:, c:size - c]).pow(2).mean(1))
        return torch.stack(errs, 1), torch.stack(incs, 1)

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    vidx = torch.arange(n_tr, n_all).repeat_interleave(16).to(device)
    vj = torch.randint(0, Nt - unroll, (vidx.numel(),), generator=g).to(device)
    vp = torch.randint(0, Nx, (vidx.numel(),), generator=g).to(device)

    def validate():
        stp.eval()
        with torch.no_grad():
            e, c = rollout(vidx, vj, vp)
            v = float((e / (c + 0.1 * c.mean(0, keepdim=True) + 1e-12)).mean().sqrt())
        stp.train()
        return v

    per_epoch = max(n_tr * 8 // batch, 1)
    first = 0
    if ckpt and os.path.exists(ckpt):
        first, best, best_state = load_partial(ckpt, stp, opt, sched)
        if verbose:
            print(f"      {kind} stepper: resumed from {ckpt} after epoch {first}", flush=True)
    for ep in range(first, epochs):
        tot = 0.0
        for _ in range(per_epoch):
            idx = torch.randint(0, n_tr, (batch,), device=device)
            j0 = torch.randint(0, Nt - unroll, (batch,), device=device)
            pos = torch.randint(0, Nx, (batch,), device=device)
            e, _ = rollout(idx, j0, pos)
            loss = e.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            v = validate()
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in stp.state_dict().items() if torch.is_tensor(t)}
        if verbose and (ep == 0 or (ep + 1) % max(epochs // 10, 1) == 0):
            print(f"      {kind} stepper epoch {ep + 1}/{epochs}: train loss {tot / per_epoch:.3e} | "
                  f"held-out increment error {validate():.3f} (best {best:.3f})", flush=True)
        if ckpt and (ep + 1) % ckpt_every == 0 and ep + 1 < epochs:
            save_partial(ckpt, ep, stp, opt, sched, best, best_state)
    if best_state is not None:
        stp.load_state_dict(best_state)
        if verbose:
            print(f"      {kind} stepper: returning the best held-out state (increment error {best:.3f})", flush=True)
    return stp
