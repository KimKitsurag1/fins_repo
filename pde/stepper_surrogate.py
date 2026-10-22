"""stepper_surrogate -- time-stepping solution operator for FINS.

u(t) -> u(t + dt) as u + F(u), F a 1-D FNO without a coordinate channel,
so every step is translation-equivariant by construction; the trajectory
is the rollout, and jvp differentiates the product of step Jacobians
exactly. exp28 (small Burgers): 5-7x lower trajectory error than the
one-shot u0 -> u(., t) FNO and a more faithful Jacobian on the non-symmetry
u d_u, but NOT on Galilean (residual ~1.6 at 16, 128 and 512 training ICs).
Since the probe does not predict discovery (tanv3), the decisive test is a
FINS run through it. Fixed-grid equations only (Burgers, KdV): the step is
learned at the stored spacing, which varies per sample on multi-scale heat.
"""

import os

import torch
import torch.nn as nn
from neuralop.models import FNO


class Stepper(nn.Module):
    def __init__(self, modes=24, width=48, layers=4, positional_embedding="grid"):
        super().__init__()
        # neuralop's default "grid" embedding feeds x in [0, 1] as an input channel, which breaks
        # shift equivariance on the torus; None keeps the operator translation-equivariant
        self.f = FNO(n_modes=(modes,), hidden_channels=width, in_channels=1,
                     out_channels=1, n_layers=layers, positional_embedding=positional_embedding)

    def forward(self, u):
        return u + self.f(u)


class RolloutOperator(nn.Module):
    """(u0, x_coords, t_coords) -> [B, 1, Nx, Nt], the signature FINS uses."""

    def __init__(self, stepper, nt):
        super().__init__()
        self.step, self.nt = stepper, nt

    def forward(self, u0, x_coords=None, t_coords=None):
        u = u0 if u0.dim() == 3 else u0.unsqueeze(1)
        out = [u]
        for _ in range(self.nt - 1):
            u = self.step(u)
            out.append(u)
        return torch.stack(out, dim=-1)


def _increment_terms(stp, sols, idx, j0, unroll):
    """Unrolled errors and increments: per sample and step k, ||pred_k -
    true_k||^2 and ||true_k - true_0||^2."""
    u0 = sols[idx, :, :, j0]
    u = u0
    errs, incs = [], []
    for k in range(1, unroll + 1):
        u = stp(u)
        tk = sols[idx, :, :, j0 + k]
        errs.append((u - tk).pow(2).mean(dim=(1, 2)))
        incs.append((tk - u0).pow(2).mean(dim=(1, 2)))
    return torch.stack(errs, 1), torch.stack(incs, 1)


def _relative(errs, incs, floor=0.1):
    """Error relative to the increment, with a floor at a fraction of the
    batch-mean increment so near-stationary windows cannot dominate."""
    den = incs + floor * incs.mean(dim=0, keepdim=True) + 1e-12
    return errs / den


def save_partial(path, ep, model, opt, sched, best, best_state):
    """Training state after epoch ep (atomic write): resuming reproduces an uninterrupted run."""
    state = dict(epoch=ep, model={k: t for k, t in model.state_dict().items() if torch.is_tensor(t)},
                 opt=opt.state_dict(), sched=sched.state_dict(), best=best, best_state=best_state,
                 rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_partial(path, model, opt, sched):
    """Restore a save_partial state; returns (first epoch to run, best, best_state)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"] + 1, state["best"], state["best_state"]


def train_stepper(train_sol, device, epochs=400, unroll=4, lr=2e-3,
                  batch=64, seed=0, verbose=True, loss="abs", val_frac=0.1,
                  eval_every=10, ckpt=None, ckpt_every=10, **arch):
    """Multi-step (unrolled) training on windows of the stored trajectories.

    loss="abs": MSE on u. loss="incr": error of the
    k-step increment relative to the increment itself. The symmetry criterion
    compares increments, and on slow dynamics an accurate trajectory can
    still carry large increment error (heat: 0.16 percent of u but ~40-70
    percent of the one-step increment under loss="abs").
    A fixed held-out set of windows is scored with the relative increment
    error every `eval_every` epochs and the best state is returned; the
    per-epoch training loss uses random windows and is too noisy for model
    selection. ckpt: path of a partial state written every ckpt_every
    epochs and resumed from if present (same result as an uninterrupted run).
    """
    torch.manual_seed(seed)
    sols = train_sol.to(device)
    n_all, _, nx, nt = sols.shape
    n_val = max(1, int(round(val_frac * n_all))) if val_frac > 0 else 0
    tr, va = sols[:n_all - n_val], sols[n_all - n_val:]
    n = tr.shape[0]
    stp = Stepper(**arch).to(device)
    opt = torch.optim.Adam(stp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    starts = torch.arange(0, nt - unroll, max((nt - unroll) // 16, 1), device=device)
    best, best_state = float("inf"), None

    def validate():
        if n_val == 0:
            return float("nan")
        stp.eval()
        with torch.no_grad():
            vals = []
            for j in starts.tolist():
                idx = torch.arange(n_val, device=device)
                e, c = _increment_terms(stp, va, idx, torch.full_like(idx, j), unroll)
                vals.append(_relative(e, c))
            v = torch.cat(vals).mean().sqrt().item()
        stp.train()
        return v

    first = 0
    if ckpt and os.path.exists(ckpt):
        first, best, best_state = load_partial(ckpt, stp, opt, sched)
        if verbose:
            print(f"      stepper: resumed from {ckpt} after epoch {first}", flush=True)
    for ep in range(first, epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            j0 = torch.randint(0, nt - unroll, (idx.numel(),), device=device)
            e, c = _increment_terms(stp, tr, idx, j0, unroll)
            l = _relative(e, c).mean() if loss == "incr" else e.mean()
            opt.zero_grad()
            l.backward()
            opt.step()
            tot += l.item() * idx.numel()
        sched.step()
        last = ep + 1 == epochs
        if (ep + 1) % eval_every == 0 or last:
            v = validate()
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in stp.state_dict().items()
                              if torch.is_tensor(t)}
        if verbose and (ep == 0 or (ep + 1) % max(epochs // 10, 1) == 0):
            v = validate()
            print(f"      stepper epoch {ep + 1}/{epochs}: train {loss} loss "
                  f"{tot / n:.3e} | held-out increment error {v:.3f} "
                  f"(best {best:.3f})", flush=True)
        if ckpt and (ep + 1) % ckpt_every == 0 and not last:
            save_partial(ckpt, ep, stp, opt, sched, best, best_state)
    if best_state is not None:
        # tensors only: neuralop's state_dict carries a '_metadata' entry that
        # its load_state_dict rejects
        stp.load_state_dict(best_state)
        if verbose:
            print(f"      stepper: returning the best held-out state "
                  f"(increment error {best:.3f})", flush=True)
    return stp
