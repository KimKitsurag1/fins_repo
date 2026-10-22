"""exp72: does a symmetry found by FINS make an emulator generalise?

KS, KdV and Benjamin-Ono on the torus, seeds 42-44. Every arm trains the same stepper (FNO, 4-step unrolled MSE, the
held-out increment error picks the best state) on the protocol's 'mixed' data (constant backgrounds within
[-0.5, 0.5]); the arms differ in what is done to the training windows:
  'none': nothing;
  'fins', 'oracle', 'lienlsd', 'shift': every window, with probability P_AUG, is moved by a random unit element of a
     set of fields of W at a parameter uniform in [-S, S] (S: the oracle's boost reaches background B_MAX). W: the
     class's fields xi(t) d_x + phi(t) d_u (free of x and u, tau = 0); they act on a periodic state at time t exactly,
     u(x) -> u(x - s xi(t)) + s phi(t);
     'fins': the part inside W of the span FINS weak returned through the learned operators (the paper's rows,
     own count): its principal directions with cosine >= COS_MIN in the scoring metric, projected onto W;
     'lienlsd': the same for LieNLSD as published (own count);
     'oracle': d_x and the Galilean boost t d_x + PHI0 d_u;
     'shift': d_x and d_u (a constant added without its drift: the naive augmentation);
  'wide': nothing, on the 'wide' data (the same draws, backgrounds within [-3, 3]): the reference of real data.
Scores: one-step and 16-step errors relative to the state's deviation from its mean, over STARTS and the ten evaluation
trajectories (data seed 1000 + seed, zero background: in distribution) and over their exact Galilean boosts to the
backgrounds BACKGROUNDS (out of distribution).
'spans' reads the 'fins' and 'lienlsd' spans from the protocol's files under FINS_RESULTS_DIR and saves their parts
inside W to results/utility (in the repository; 'train' reads them there).
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp42_tables as e42                                 # noqa: E402
from stepper_surrogate import Stepper, _relative, save_partial, load_partial   # noqa: E402
from window_stepper import to_float64                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EQS, SEEDS = ("ks", "kdv", "bo"), (42, 43, 44)
BASE, WIDE = "mixed", "wide"                               # training data of every arm but 'wide', and of 'wide'
PHI0 = {"ks": 1.0, "kdv": 1.0 / 6.0, "bo": 1.0}            # Galilean boost (t, 0, PHI0): x -> x + s t, u -> u + PHI0 s
ARMS = ("none", "fins", "oracle", "lienlsd", "shift", "wide")
N_TRAIN, EPOCHS, UNROLL, BATCH, LR = 512, 300, 4, 64, 2e-3
P_AUG, B_MAX, COS_MIN = 0.5, 3.0, 0.99
BACKGROUNDS = (-2.0, -1.0, 1.0, 2.0)
STARTS, HORIZON = (8, 40, 72, 104), 16
N_EVAL = 10
# the paper's files (clean, seeds 42-44): FINS weak through the learned operators; LieNLSD as published (KdV: exp41
# with the noise study's settings at noise 0, as for BO and KS)
FINS_FILE = {"ks": "ks_runs/clean", "kdv": "noise_weak_runs/clean", "bo": "bo_runs/clean"}
LIENLSD_FILE = {"ks": "ks_runs/clean", "kdv": "utility_spans", "bo": "bo_runs/clean"}


_GRID = {}


def grid(pde):
    """L, x, t, T (cached: KdV's grid comes from a solve)."""
    if pde not in _GRID:
        T = e39.horizon(pde)
        L, x, t = e39.grid(pde, e31.ExperimentConfig(), 128, 128, T)
        _GRID[pde] = float(L), np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64), float(T)
    return _GRID[pde]


def training_set(pde, seed, n, cache, kind=BASE):
    path = os.path.join(cache, f"{pde}_{kind}_train_s{seed}_n{n}.npy")
    if os.path.exists(path):
        return np.load(path)
    _, x, _, T = grid(pde)
    tr = e39.training_data(pde, n, seed, kind, x, T, 128, 0.002).astype(np.float32)
    os.makedirs(cache, exist_ok=True)
    np.save(path + f".{os.getpid()}.npy", tr)
    os.replace(path + f".{os.getpid()}.npy", path)
    return tr


def evaluation_set(pde, seed):
    """The protocol's evaluation trajectories (data seed 1000 + seed; zero background), [N, Nx, Nt], and times."""
    _, x, t, _ = grid(pde)
    u, _, ts = e39.evaluation_data(pde, "periodic", N_EVAL, 1000 + seed, x, t, 0.002)
    return np.asarray(u, dtype=np.float64), np.asarray(ts, dtype=np.float64)


def reference(pde, seed):
    """FINS's class coordinates on the evaluation data (as exp49 sets them)."""
    L, x, _, _ = grid(pde)
    u, ts = evaluation_set(pde, seed)
    d = 2
    return dict(u=u, xs=np.tile(x, (len(u), 1)), ts=ts, Lw=L, x_offset=np.zeros(len(u)), T_max=float(ts[:, -1].max()),
                sigma=float(u.std()), mons=np.array([m for m in itertools.product(range(d + 1), repeat=3)
                                                    if sum(m) <= d]))


def scored(z):
    """The scoring metric's region and weights."""
    Nx = z["u"].shape[1]
    frac = np.arange(Nx) / Nx
    return (frac >= 0.2) & (frac <= 0.8), np.array([1 / z["Lw"], 1 / z["ts"][:, -1].max(), 1 / z["u"].std()])


def class_vectors(z):
    """The class's basis fields as scored vectors [3 nm, dim]."""
    keep, w = scored(z)
    return e42.vectors(e42.class_fields(z), keep, w, 8)


def metric(M):
    """Gram matrix of the class's basis fields in the scoring metric [3 nm, 3 nm]."""
    return M @ M.T / M.shape[1]


def w_rows(z):
    """Rows of W: xi and phi monomials free of X and U; tau excluded."""
    nm = len(z["mons"])
    return [q * nm + j for q in (0, 2) for j, (a, _, c) in enumerate(z["mons"]) if a == 0 and c == 0]


def g_orth(B, G):
    """G-orthonormal basis of the columns of B."""
    S = B.T @ G @ B
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-12 * lam.max()
    return B @ V[:, keep] / np.sqrt(lam[keep])


def torus_part(B, z, G):
    """Principal directions of span(B) inside W with cosine >= COS_MIN, projected onto W and G-normalised: [3 nm, m],
    and every principal cosine."""
    rows = w_rows(z)
    E = np.zeros((G.shape[0], len(rows)))
    E[rows, np.arange(len(rows))] = 1.0
    Qb, Qw = g_orth(B, G), g_orth(E, G)
    U, s, _ = np.linalg.svd(Qb.T @ G @ Qw)
    V = Qb @ U[:, s >= COS_MIN]
    V = Qw @ (Qw.T @ G @ V)
    return (g_orth(V, G) if V.shape[1] else V), s


def oracle_span(pde, z):
    nm = len(z["mons"])
    mons = [tuple(m) for m in z["mons"]]
    one, T1 = mons.index((0, 0, 0)), mons.index((0, 1, 0))
    B = np.zeros((3 * nm, 2))
    B[one, 0] = 1.0                                         # d_x
    B[T1, 1], B[2 * nm + one, 1] = z["T_max"], PHI0[pde]    # t d_x + PHI0 d_u
    return B


def shift_span(z):
    nm = len(z["mons"])
    one = [tuple(m) for m in z["mons"]].index((0, 0, 0))
    B = np.zeros((3 * nm, 2))
    B[one, 0], B[2 * nm + one, 1] = 1.0, 1.0                # d_x, d_u
    return B


def boost_element(V, z, pde):
    """The element of span(V) with xi(0) = 0 and phi(0) = PHI0: its xi = a1 t + a2 t^2, phi = PHI0 + b1 t + b2 t^2
    (the boost: a1 = 1, the rest 0)."""
    nm = len(z["mons"])
    mons = [tuple(m) for m in z["mons"]]
    one, T1, T2 = mons.index((0, 0, 0)), mons.index((0, 1, 0)), mons.index((0, 2, 0))
    a = np.linalg.lstsq(np.stack([V[one], V[2 * nm + one]]), np.array([0.0, PHI0[pde]]), rcond=None)[0]
    e = V @ a
    T = z["T_max"]
    return dict(a1=e[T1] / T, a2=e[T2] / T ** 2, b1=e[2 * nm + T1] / T, b2=e[2 * nm + T2] / T ** 2)


def spans_file(pde, seed):
    return os.path.join(HERE, "results", "utility", f"spans_{pde}_s{seed}.npz")


class Move:
    """Exact action of fields of W on periodic states at their times: u(x) -> u(x - s xi(t)) + s phi(t)."""

    def __init__(self, z, L, Nx, device):
        self.idx = [(j, int(b)) for j, (a, b, c) in enumerate(z["mons"]) if a == 0 and c == 0]
        self.nm, self.T_max = len(z["mons"]), float(z["T_max"])
        self.ik = 1j * torch.fft.fftfreq(Nx, d=L / Nx, device=device, dtype=torch.float64) * 2 * np.pi

    def __call__(self, u, t, coef, s):
        """u [B, Nx] at times t [B], fields coef [B, 3 nm] (physical components), parameters s [B]."""
        T = t / self.T_max
        xi = sum(coef[:, j] * T ** b for j, b in self.idx)
        phi = sum(coef[:, 2 * self.nm + j] * T ** b for j, b in self.idx)
        F = torch.fft.fft(u.to(torch.float64)) * torch.exp(-self.ik[None] * (s * xi)[:, None])
        return torch.fft.ifft(F).real + (s * phi)[:, None]


def boost_exact(u, ts, b, pde, L):
    """Exact Galilean boost of trajectories [N, Nx, Nt] at times ts [N, Nt] to background b."""
    s = b / PHI0[pde]
    k = np.fft.fftfreq(u.shape[1], d=L / u.shape[1]) * 2 * np.pi
    F = np.fft.fft(u, axis=1)
    return np.fft.ifft(F * np.exp(-1j * k[None, :, None] * s * ts[:, None, :]), axis=1).real + b


def augment_scale(pde, z, G):
    """S: the parameter at which the G-normalised oracle boost reaches background B_MAX (basis-free)."""
    b = oracle_span(pde, z)[:, 1]
    return B_MAX * np.sqrt(b @ G @ b) / PHI0[pde]


def train(tr, t, L, z, V, S, device, seed, ckpt, epochs, verbose=True):
    """The objective of stepper_surrogate.train_stepper (4-step unrolled MSE on u, held-out increment error for the
    best state) with every training window moved, with probability P_AUG, by a random unit element of span(V) at a
    parameter in [-S, S]."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 7919)
    sols = torch.tensor(tr, dtype=torch.float32, device=device)[:, None]
    n_all, _, nx, nt = sols.shape
    n_val = max(1, int(round(0.1 * n_all)))
    trs, va = sols[:n_all - n_val], sols[n_all - n_val:]
    n = trs.shape[0]
    stp = Stepper(modes=64, positional_embedding=None).to(device)
    opt = torch.optim.Adam(stp.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tt = torch.tensor(t, dtype=torch.float64, device=device)
    Vt = torch.tensor(V, dtype=torch.float64, device=device) if V is not None and V.shape[1] else None
    move = Move(z, L, nx, device) if Vt is not None else None
    starts = list(range(0, nt - UNROLL, max((nt - UNROLL) // 16, 1)))

    def unrolled(win):
        u, errs, incs = win[..., 0], [], []
        for m in range(1, UNROLL + 1):
            u = stp(u)
            errs.append((u - win[..., m]).pow(2).mean(dim=(1, 2)))
            incs.append((win[..., m] - win[..., 0]).pow(2).mean(dim=(1, 2)))
        return torch.stack(errs, 1), torch.stack(incs, 1)

    def validate():
        stp.eval()
        with torch.no_grad():
            v = torch.cat([_relative(*unrolled(va[..., j:j + UNROLL + 1])) for j in starts]).mean().sqrt().item()
        stp.train()
        return v

    first, best, best_state = 0, float("inf"), None
    if ckpt and os.path.exists(ckpt):
        first, best, best_state = load_partial(ckpt, stp, opt, sched)
        gen.manual_seed(seed + 7919 + first)
    for ep in range(first, epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            j0 = torch.randint(0, nt - UNROLL, (idx.numel(),), device=device)
            jj = j0[:, None] + torch.arange(UNROLL + 1, device=device)[None, :]
            win = sols[idx[:, None], 0, :, jj].permute(0, 2, 1)[:, None]                            # [B, 1, Nx, U+1]
            if move is not None:
                sel = torch.rand(idx.numel(), generator=gen, device=device) < P_AUG
                if sel.any():
                    m, q = int(sel.sum()), UNROLL + 1
                    a = torch.randn(m, Vt.shape[1], generator=gen, device=device, dtype=torch.float64)
                    coef = (a / a.norm(dim=1, keepdim=True)) @ Vt.T
                    s = (2 * torch.rand(m, generator=gen, device=device, dtype=torch.float64) - 1) * S
                    w = win[sel, 0].permute(0, 2, 1).reshape(m * q, nx)                            # [m (U+1), Nx]
                    moved = move(w, tt[jj[sel]].reshape(m * q), coef.repeat_interleave(q, 0), s.repeat_interleave(q, 0))
                    win = win.clone()
                    win[sel, 0] = moved.reshape(m, q, nx).permute(0, 2, 1).to(win.dtype)
            e, _ = unrolled(win)
            loss = e.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        sched.step()
        last = ep + 1 == epochs
        if (ep + 1) % 10 == 0 or last:
            v = validate()
            if v < best:
                best, best_state = v, {k: q.detach().clone() for k, q in stp.state_dict().items() if torch.is_tensor(q)}
            if verbose and ((ep + 1) % max(epochs // 10, 10) == 0 or last):
                print(f"      epoch {ep + 1}/{epochs}: train loss {tot / n:.3e} | held-out increment error {v:.4f} "
                      f"(best {best:.4f})", flush=True)
        if ckpt and (ep + 1) % 10 == 0 and not last:
            save_partial(ckpt, ep, stp, opt, sched, best, best_state)
    if best_state is not None:                  # every validation NaN: the last state
        stp.load_state_dict(best_state)
    return stp, best


def stepper_file(runs, pde, seed, arm):
    return os.path.join(runs, f"S_{pde}_s{seed}_{arm}.pt")


def load_stepper(path):
    stp = Stepper(modes=64, positional_embedding=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    stp.load_state_dict({k: v for k, v in sd.items() if torch.is_tensor(v)})
    return to_float64(stp).eval()


def errors(stp, u, device):
    """Mean one-step and HORIZON-step errors relative to the state's deviation from its mean, over STARTS."""
    one, far = [], []
    with torch.no_grad():
        for j0 in STARTS:
            v = torch.tensor(u[:, None, :, j0], device=device)
            for m in range(1, HORIZON + 1):
                v = stp(v)
                ref = u[:, :, j0 + m]
                e = np.linalg.norm(v[:, 0].cpu().numpy() - ref, axis=1) / np.linalg.norm(
                    ref - ref.mean(1, keepdims=True), axis=1)
                if m == 1:
                    one.append(e)
                if m == HORIZON:
                    far.append(e)
    return float(np.mean(one)), float(np.mean(far))


def check_reference(f, z, path):
    for key in ("Lw", "T_max", "sigma"):
        if not np.isclose(float(f[key]), float(z[key]), rtol=1e-6):
            raise SystemExit(f"{path}: {key} {float(f[key])} differs from the reference {float(z[key])}")


def cmd_spans(args):
    """The 'fins' and 'lienlsd' spans' parts inside W from the protocol's files -> results/utility."""
    R = os.environ.get("FINS_RESULTS_DIR", "results")
    for pde in ([args.pde] if args.pde else EQS):
        for seed in SEEDS:
            z = reference(pde, seed)
            M = class_vectors(z)
            G = metric(M)
            Vo, _ = torus_part(oracle_span(pde, z), z, G)
            Qo = g_orth(Vo, G)
            out = dict(Lw=z["Lw"], T_max=z["T_max"], sigma=z["sigma"], mons=z["mons"])
            p = os.path.join(R, FINS_FILE[pde], f"weak-learned_{pde}_s{seed}.npz")
            f = np.load(p, allow_pickle=True)
            check_reference(f, z, p)
            if not np.array_equal(f["mons"], z["mons"]):
                raise SystemExit(f"{p}: another class")
            spans = {"fins": (f["C"][:, :int(f["nown"])], int(f["nown"]), 0.0, p)}
            p = os.path.join(R, LIENLSD_FILE[pde], f"lienlsd-f_{pde}_s{seed}.npz")
            f = np.load(p, allow_pickle=True)
            if f["u"].shape != z["u"].shape or not np.allclose(f["u"], z["u"], atol=1e-6 * np.abs(z["u"]).max()):
                raise SystemExit(f"{p}: evaluation trajectories differ from the reference")
            n_own = int(f["nown_LieNLSD_F"])
            keep, w = scored(z)
            D = e42.vectors(np.moveaxis(f["fields_LieNLSD_F"].astype(np.float64), 0, 2)[:n_own], keep, w, 8)
            C, *_ = np.linalg.lstsq(M.T, D.T, rcond=None)                   # the generators in FINS's class
            rel = float(np.linalg.norm(M.T @ C - D.T) / np.linalg.norm(D))
            spans["lienlsd"] = (C, n_own, rel, p)
            for arm, (B, m, rel, p) in spans.items():
                V, cos = torus_part(B, z, G)
                pc = np.linalg.svd(Qo.T @ G @ V, compute_uv=False) if V.shape[1] else np.zeros(0)
                be = boost_element(V, z, pde) if V.shape[1] >= 2 else {}
                out.update({f"V_{arm}": V, f"cos_{arm}": cos, f"own_{arm}": m, f"fit_{arm}": rel,
                            f"to_oracle_{arm}": pc, f"boost_{arm}": json.dumps(be), f"source_{arm}": p})
                print(f"[{pde} s{seed} {arm}] own count {m}, class fit residual {rel:.1e}, part in W "
                      f"{V.shape[1]} (cosines {' '.join(f'{c:.4f}' for c in cos)}), cosines to the oracle's "
                      f"{' '.join(f'{c:.4f}' for c in pc)}"
                      + (f"; boost element xi = {be['a1']:.4f} t {be['a2']:+.2e} t^2, phi = PHI0 {be['b1']:+.2e} t "
                         f"{be['b2']:+.2e} t^2" if be else ""), flush=True)
            os.makedirs(os.path.dirname(spans_file(pde, seed)), exist_ok=True)
            np.savez(spans_file(pde, seed), **out)


def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L, _, t, _ = grid(args.pde)
    z = reference(args.pde, args.seed)
    G = metric(class_vectors(z))
    out = stepper_file(args.runs, args.pde, args.seed, args.arm)
    if os.path.exists(out):
        print(f"exists: {out}")
        return
    kind = WIDE if args.arm == "wide" else BASE
    tr = training_set(args.pde, args.seed, args.n_train, args.cache, kind)
    info = dict(pde=args.pde, seed=args.seed, arm=args.arm, n_train=args.n_train, epochs=args.epochs, kind=kind)
    V, S = None, 0.0
    if args.arm not in ("none", "wide"):
        S = augment_scale(args.pde, z, G)
        if args.arm in ("fins", "lienlsd"):
            p = args.spans or spans_file(args.pde, args.seed)
            f = np.load(p, allow_pickle=True)
            check_reference(f, z, p)
            V, cos = f[f"V_{args.arm}"], f[f"cos_{args.arm}"]
        else:
            V, cos = torus_part(oracle_span(args.pde, z) if args.arm == "oracle" else shift_span(z), z, G)
        info.update(dim=int(V.shape[1]), cosines=[float(c) for c in cos], S=float(S))
        print(f"[{args.pde} s{args.seed} {args.arm}] part in W {V.shape[1]} (cosines "
              + " ".join(f"{c:.4f}" for c in cos) + f"), S {S:.4g}", flush=True)
        if not V.shape[1]:
            print("    empty: trained as 'none'", flush=True)
    t0 = time.time()
    stp, best = train(tr, t, L, z, V, S, dev, args.seed, out + ".partial", args.epochs)
    sd = {k: v.detach().cpu() for k, v in stp.state_dict().items() if torch.is_tensor(v)}
    torch.save(sd, out)
    if os.path.exists(out + ".partial"):
        os.remove(out + ".partial")
    info.update(best_val=best, seconds=time.time() - t0, device=dev)
    with open(out.replace(".pt", ".json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"saved {out} ({time.time() - t0:.0f} s)", flush=True)


def cmd_evaluate(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L, _, _, _ = grid(args.pde)
    u, ts = evaluation_set(args.pde, args.seed)
    res = {}
    for arm in ARMS:
        p = stepper_file(args.runs, args.pde, args.seed, arm)
        if not os.path.exists(p):
            continue
        stp = load_stepper(p).to(dev)
        r = {"in": errors(stp, u, dev)}
        for b in BACKGROUNDS:
            r[f"{b:+.2f}"] = errors(stp, boost_exact(u, ts, b, args.pde, L), dev)
        oo = [r[f"{b:+.2f}"] for b in BACKGROUNDS]
        r["out"] = (float(np.mean([o[0] for o in oo])), float(np.mean([o[1] for o in oo])))
        res[arm] = r
        print(f"[{args.pde} s{args.seed} {arm:8s}] in: one-step {r['in'][0]:.2e}, {HORIZON}-step {r['in'][1]:.2e} | "
              f"out: one-step {r['out'][0]:.2e}, {HORIZON}-step {r['out'][1]:.2e}", flush=True)
    with open(os.path.join(args.runs, f"eval_{args.pde}_s{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


def cmd_summary(args):
    """16-step errors of every arm (eval_<pde>_s<seed>.json in --runs), the registered verdicts UT1-UT7 ->
    results/utility/summary.{md,json} and verdicts.txt."""
    res, dims = {}, {}
    for pde in EQS:
        for seed in SEEDS:
            res[(pde, seed)] = json.load(open(os.path.join(args.runs, f"eval_{pde}_s{seed}.json")))
            info = json.load(open(stepper_file(args.runs, pde, seed, "fins").replace(".pt", ".json")))
            dims[(pde, seed)] = info["dim"]

    def e(pde, seed, arm, where):
        return res[(pde, seed)][arm]["in" if where == "ID" else "out"][1]

    lines = ["| equation, seed | " + " | ".join(f"{a} ID / OOD" for a in ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for (pde, seed) in res:
        lines.append(f"| {pde} {seed} | " + " | ".join(f"{e(pde, seed, a, 'ID'):.2e} / {e(pde, seed, a, 'OOD'):.2e}"
                                                      for a in ARMS) + " |")
    checks = {
        "UT1": ("none: OOD >= 3 ID", [(p, s, e(p, s, "none", "OOD") / e(p, s, "none", "ID"), ">=", 3.0)
                                     for p, s in res]),
        "UT2": ("oracle OOD <= 1.5 wide OOD", [(p, s, e(p, s, "oracle", "OOD") / e(p, s, "wide", "OOD"), "<=", 1.5)
                                              for p, s in res]),
        "UT3": ("fins OOD <= 0.5 none OOD (a part in W)", [(p, s, e(p, s, "fins", "OOD") / e(p, s, "none", "OOD"),
                                                           "<=", 0.5) for p, s in res if dims[(p, s)]]),
        "UT4": ("fins OOD <= 1.5 oracle OOD (KS, KdV)", [(p, s, e(p, s, "fins", "OOD") / e(p, s, "oracle", "OOD"),
                                                         "<=", 1.5) for p, s in res if p in ("ks", "kdv")]),
        "UT5": ("shift OOD >= none OOD", [(p, s, e(p, s, "shift", "OOD") / e(p, s, "none", "OOD"), ">=", 1.0)
                                         for p, s in res]),
        "UT6": ("fins, oracle ID <= 2.5 none ID", [(p, s, max(e(p, s, "fins", "ID"), e(p, s, "oracle", "ID"))
                                                    / e(p, s, "none", "ID"), "<=", 2.5) for p, s in res]),
        "UT7": ("lienlsd OOD <= 1.5 oracle OOD (KS); >= 0.8 none OOD (KdV, BO)",
                [(p, s, e(p, s, "lienlsd", "OOD") / e(p, s, "oracle", "OOD"), "<=", 1.5) for p, s in res if p == "ks"]
                + [(p, s, e(p, s, "lienlsd", "OOD") / e(p, s, "none", "OOD"), ">=", 0.8) for p, s in res if p != "ks"]),
    }
    verdicts, out = [], {"errors": {f"{p} {s}": r for (p, s), r in res.items()}, "verdicts": {}}
    for name, (text, rows) in checks.items():
        ok = [r >= b if op == ">=" else r <= b for _, _, r, op, b in rows]
        verdicts.append(f"({name}) {'MET' if all(ok) else 'NOT MET'} ({sum(ok)}/{len(ok)}): {text}; ratios "
                        + ", ".join(f"{p} {s} {r:.3g}" for p, s, r, _, _ in rows))
        out["verdicts"][name] = dict(met=bool(all(ok)), n_ok=int(sum(ok)), n=len(ok),
                                     ratios={f"{p} {s}": float(r) for p, s, r, _, _ in rows})
    d = os.path.join(HERE, "results", "utility")
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("16-step errors relative to the state's deviation from its mean; ID: evaluation trajectories, "
                "OOD: their exact Galilean boosts to backgrounds -2, -1, 1, 2 (mean)\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(d, "verdicts.txt"), "w") as f:
        f.write("\n".join(verdicts) + "\n")
    print("\n".join(lines + [""] + verdicts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["spans", "train", "evaluate", "summary"])
    ap.add_argument("--pde", choices=EQS, help="required for train and evaluate; spans: every equation by default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default="none", choices=ARMS)
    ap.add_argument("--spans", default=None, help="train: spans file of 'fins' and 'lienlsd' (default results/utility)")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "utility_runs"))
    ap.add_argument("--cache", default=os.environ.get("FINS_DATA_CACHE",
                                                      os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"),
                                                                   "data_cache")))
    args = ap.parse_args()
    if args.cmd in ("train", "evaluate") and not args.pde:
        ap.error("--pde is required")
    os.makedirs(args.runs, exist_ok=True)
    {"spans": cmd_spans, "train": cmd_train, "evaluate": cmd_evaluate, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
