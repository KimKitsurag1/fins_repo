"""Experiment 60 -- anisotropic nonlocal dissipation in 2D: a stretched rotation and a fractional scaling read from the
operator without the equation (README_EXPERIMENTS.md; development 08.10).

Equation aniso2d.py: u_t = div_A(u grad_A u) - nu (-div_A grad_A)^(a/2) u, grad_A = (d_x, eps d_y), on exp45's box;
algebra d_x, d_y, d_t, -y / eps d_x + eps x d_y, x d_x + y d_y + a t d_t + (2 - a) u d_u. FINS: exp45's polynomial
class (60 fields, degree 2 in x, y, t, u) through the exact operator's tangent, the relative pencil on the training
trajectories, scores and readouts on held-out trajectories in characteristic space:
  eps-hat   the direction of the plane (y d_x, x d_y) deepest in the span, reduced against the translations: the
            coefficients (c1, c2) give eps-hat = sqrt(-c2 / c1);
  a-hat     the direction of (x d_x + y d_y, t d_t, u d_u) deepest in the span, reduced against the translations and the
            rotation read: (1, a-hat, c-hat), c-hat = 2 - a-hat for the true scaling.
Baselines (regress; wsindy2d.py): weak SINDy in 2D on the same training trajectories with a local library, and with
nonlocal completions whose order (and anisotropy) the residual chooses; each identified model's algebra by its
structure (stretched rotation: an elliptic symbol of the nonlinearity's anisotropy; scaling: homogeneity).

Run:  python exp60_aniso2d.py fins --a 1.5 --seed 7 [--data-seed 100 --eval-seed 101] --save <stem>
      python exp60_aniso2d.py check --a 1.5
"""

import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import aniso2d as an                                       # noqa: E402
import exp42_tables as e42                                 # noqa: E402
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp56_wsindy as e56                                 # noqa: E402
import exp58_symmetry_library as e58                       # noqa: E402
import method_wsindy2d as mw2                              # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
GATE = 0.5
MAX_OWN = 12
POLYS, MAX_D = (0, 1, 2, 3), 4
BETAS = np.round(np.arange(0.5, 4.0 + 1e-9, 0.05), 4)
EPSS = np.round(np.arange(0.3, 1.0 + 1e-9, 0.05), 4)
NEST = 1.25
EPS_TOL, DEG_TOL = 0.02, 0.1


def own_count(lam, k, floor=1e-12):
    """The largest of the first k + 4 ratios of consecutive eigenvalues floored at `floor` (exp49's rule with its own
    floor, 1e-6, and up to 17 eigenvalues, gives 5-6 on exp60's runs; README entry of 09.10)."""
    lf = np.maximum(np.asarray(lam)[:min(k + 4, MAX_OWN) + 1], floor)
    return int(np.argmax(lf[1:] / lf[:-1])) + 1


def readouts(probe, C, eps, a):
    """eps-hat, a-hat, c-hat and containments of a span C [60, m] on the probe (characteristic space)."""
    _, Qc = probe.poly(C)
    D = np.where(np.isfinite(Qc), Qc, 0.0)
    trans = probe.poly(an.true_coefficients(eps, a)[:, :3])[1]
    Qt = e42.orth(trans)

    def red(v):
        return v - Qt @ (Qt.T @ v)

    def char(entries):
        return probe.poly(an.basis_vector(entries)[:, None])[1][0]

    P = np.stack([red(char([(0, (0, 1, 0, 0), 1.0)])), red(char([(1, (1, 0, 0, 0), 1.0)]))], 1)   # y d_x, x d_y
    lam_r, V, _ = e58.principal(D, P)
    c1, c2 = V[0, 0], V[1, 0]
    eps_hat = float(np.sqrt(-c2 / c1)) if c1 * c2 < 0 else None
    rest = [trans[i] for i in range(3)]
    if lam_r[0] >= GATE and eps_hat is not None:
        rest.append(char([(0, (0, 1, 0, 0), -1 / eps_hat), (1, (1, 0, 0, 0), eps_hat)]))
    Qr = e42.orth(np.stack(rest))

    def red2(v):
        return v - Qr @ (Qr.T @ v)

    S = np.stack([red2(char([(0, (1, 0, 0, 0), 1.0), (1, (0, 1, 0, 0), 1.0)])), red2(char([(2, (0, 0, 1, 0), 1.0)])),
                  red2(char([(3, (0, 0, 0, 1), 1.0)]))], 1)
    lam_s, W, _ = e58.principal(D, S)
    w = W[:, 0]
    a_hat = float(w[1] / w[0]) if w[0] else None
    c_hat = float(w[2] / w[0]) if w[0] else None
    true_q = probe.poly(an.true_coefficients(eps, a))[1]
    return dict(eps_hat=eps_hat, rot_contained=float(lam_r[0]), a_hat=a_hat, c_hat=c_hat,
                scal_contained=[float(v) for v in lam_s], missing=e45.missing(D, true_q),
                novel=e45.novel(D, true_q).tolist(), m=int(C.shape[1]))


def reread(args):
    """Readouts of a saved run's spans, recomputed from its npz."""
    z = np.load(args.save + ".npz")
    g = an.Grid(args.n, args.L, args.score, eps=float(z["eps"]), a=float(z["a"]), nu=float(z["nu"]))
    dtr = e45.Data(g, torch.tensor(z["u_train"]), float(z["dt"]))
    probe = e45.Probe(e45.Data(g, torch.tensor(z["u_eval"]), float(z["dt"]), dtr.s))
    lam, C = z["lam"], z["C"]
    Ct = an.true_coefficients(g.eps, g.a)
    rq = [float(c @ z["N"] @ c / (c @ z["D"] @ c)) for c in Ct.T]
    print("algebra Rayleigh quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(an.NAMES, rq)))
    extra = {"D2 = t d_t - u d_u": an.basis_vector([(2, (0, 0, 1, 0), 1.0), (3, (0, 0, 0, 1), -1.0)]),
             "rotation": an.basis_vector([(0, (0, 1, 0, 0), -1.0), (1, (1, 0, 0, 0), 1.0)]),
             "u d_u": an.basis_vector([(3, (0, 0, 0, 1), 1.0)])}
    print("others: " + "  ".join(f"{n} {float(c @ z['N'] @ c / (c @ z['D'] @ c)):.1e}" for n, c in extra.items()))
    print("lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:8]))
    sa, sc, sq = scan_scaling(z["N"], z["D"])
    se, seq = scan_rotation(z["N"], z["D"])
    print(f"pencil scans: scaling a {sa:.3f}, c {sc:.3f} (quotient {sq:.1e}); stretched rotation eps {se:.3f} "
          f"(quotient {seq:.1e})")
    for m in (Ct.shape[1], own_count(lam, Ct.shape[1]), 6):
        r = readouts(probe, C[:, :m], g.eps, g.a)
        print(f"span of the {m} lowest: missing {r['missing']:.3f}; novel "
              + " ".join(f"{n} {v:.2f}" for n, v in zip(an.NAMES, r["novel"]))
              + f"; eps-hat {r['eps_hat']} (contained {r['rot_contained']:.3f}); a-hat {r['a_hat']}, c-hat "
              f"{r['c_hat']} (contained {r['scal_contained'][0]:.3f}, {r['scal_contained'][1]:.3f})")


def quad(N, D, V):
    """Reduced quadratic forms of the pencil on the columns of V: (V'NV, V'DV)."""
    return V.T @ N @ V, V.T @ D @ V


def scan_scaling(N, D, a_range=(0.5, 3.0), c_range=(-1.0, 2.0), step=0.01):
    """The exact scaling x d_x + y d_y + a t d_t + c u d_u with the smallest pencil quotient: (a, c, quotient); a grid
    then a tenfold finer grid around the minimum."""
    V = np.stack([an.basis_vector([(0, (1, 0, 0, 0), 1.0), (1, (0, 1, 0, 0), 1.0)]),
                  an.basis_vector([(2, (0, 0, 1, 0), 1.0)]), an.basis_vector([(3, (0, 0, 0, 1), 1.0)])], 1)
    n, d = quad(N, D, V)

    def rq(a, c):
        w = np.array([1.0, a, c])
        return np.einsum("i,ij,j->", w, n, w) / np.einsum("i,ij,j->", w, d, w)

    best = None
    for st in (step, step / 10):
        if best is None:
            (alo, ahi), (clo, chi) = a_range, c_range
        else:
            alo, ahi, clo, chi = best[0] - step, best[0] + step, best[1] - step, best[1] + step
        A, C = np.meshgrid(np.arange(alo, ahi + 1e-12, st), np.arange(clo, chi + 1e-12, st), indexing="ij")
        W = np.stack([np.ones_like(A), A, C], -1)
        q = np.einsum("...i,ij,...j->...", W, n, W) / np.einsum("...i,ij,...j->...", W, d, W)
        i = np.unravel_index(int(np.argmin(q)), q.shape)
        best = (float(A[i]), float(C[i]), float(q[i]))
    return best


def scan_rotation(N, D, e_range=(0.2, 2.0), step=0.001):
    """The stretched rotation -y / e d_x + e x d_y with the smallest pencil quotient: (e, quotient)."""
    V = np.stack([an.basis_vector([(0, (0, 1, 0, 0), 1.0)]), an.basis_vector([(1, (1, 0, 0, 0), 1.0)])], 1)
    n, d = quad(N, D, V)
    es = np.arange(e_range[0], e_range[1] + 1e-12, step)
    W = np.stack([-1 / es, es], -1)
    q = np.einsum("...i,ij,...j->...", W, n, W) / np.einsum("...i,ij,...j->...", W, d, W)
    i = int(np.argmin(q))
    return float(es[i]), float(q[i])


def fno_path(args):
    path = os.path.join(RESULTS, "aniso2d", f"fno2d_aniso_a{args.a:g}_s{args.seed}{args.fno_tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def train_fno(args, g):
    """exp45's recipe on this equation: one-step training with a two-step rollout term on fno_n trajectories of data
    seed 6000 + seed, Adam, cosine."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    u = an.trajectories(g, args.fno_n, 6000 + args.seed, args.nt, args.dt, args.nsub).float()
    model = e45.FNO2d(args.fno_width, args.fno_modes, args.fno_layers)
    inc = float((u[:, 1:] - u[:, :-1]).std())
    model.norm.copy_(torch.tensor([float(u.mean()), float(u.std()), inc]))
    model = model.to(dev)
    x0, x1, x2 = (u[:, i:u.shape[1] - 2 + i].reshape(-1, g.n, g.n) for i in range(3))
    n = x0.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.fno_lr)
    steps = args.fno_epochs * (n // args.fno_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    print(f"training FNO2d on {args.fno_n} trajectories ({n} pairs), width {args.fno_width}, modes "
          f"{args.fno_modes}, {args.fno_layers} layers, {args.fno_epochs} epochs, {dev}", flush=True)
    t0 = time.time()
    for ep in range(args.fno_epochs):
        perm, tot = torch.randperm(n), 0.0
        for b in range(n // args.fno_batch):
            i = perm[b * args.fno_batch:(b + 1) * args.fno_batch]
            a, y1, y2 = x0[i].to(dev), x1[i].to(dev), x2[i].to(dev)
            p1 = model(a)
            p2 = model(p1)
            loss = (((p1 - y1) ** 2).mean() + ((p2 - y2) ** 2).mean()) / inc ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % max(1, args.fno_epochs // 20) == 0 or ep == args.fno_epochs - 1:
            print(f"  epoch {ep}: loss {tot / (n // args.fno_batch):.3e} ({time.time() - t0:.0f} s)", flush=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=sd, cfg=dict(width=args.fno_width, modes=args.fno_modes, layers=args.fno_layers)),
               fno_path(args))
    print(f"saved {fno_path(args)}", flush=True)


def load_fno(args, g):
    path = fno_path(args)
    if not os.path.exists(path):
        if not args.train_fno_if_missing:
            raise SystemExit(f"no FNO at {path}; pass --train-fno-if-missing")
        train_fno(args, g)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = e45.FNO2d(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


def fins(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = an.Grid(args.n, args.L, args.score, eps=args.eps, a=args.a, nu=args.nu)
    t0 = time.time()
    dtr = e45.Data(g, an.trajectories(g, args.n_train, dseed, args.nt, args.dt, args.nsub), args.dt)
    dev = e45.Data(g, an.trajectories(g, args.n_eval, eseed, args.nt, args.dt, args.nsub), args.dt, dtr.s)
    print(f"[aniso2d eps {args.eps:g} a {args.a:g} nu {args.nu:g}, {args.op}] seed {s}: {args.n_train} training "
          f"trajectories (data seed {dseed}), {args.n_eval} held-out ({eseed}); {args.nt} snapshots, dt {args.dt}, K "
          f"{args.K}; u in [{float(dtr.u.min()):.3f}, {float(dtr.u.max()):.3f}]; data {time.time() - t0:.0f} s",
          flush=True)
    out = dict(config=vars(args), data_seed=dseed, eval_seed=eseed)
    if args.op == "exact":
        op = an.ExactOp(g, args.dt, args.nsub)
        d_ev = dtr
    else:
        model = load_fno(args, g.cast(torch.float32))
        op64 = e45.FNOOp(copy.deepcopy(model).double())
        for name, dd in (("training", dtr), ("held-out", dev)):
            e1, eK, per = e45.operator_errors(op64.model, dd, args.K)
            out[f"fno_error_{name}"] = [e1, eK]
            out[f"fno_error_{name}_per_trajectory"] = per.tolist()
            print(f"  FNO rollout error relative to the data increment ({name}): 1 step {e1:.4f} (per trajectory "
                  f"{per.min():.4f}-{per.max():.4f}), K steps {eK:.4f}", flush=True)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        op, d_ev = (op64, dtr) if device == "cpu" else (op64.cast(torch.float32, device),
                                                        dtr.cast(torch.float32, device))
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = e45.pencil(op, d_ev, e45.chars_poly, ptr, args.K, 2)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N, D)
    Ct = an.true_coefficients(args.eps, args.a)
    rq = [float(c @ N @ c / (c @ D @ c)) for c in Ct.T]
    k = Ct.shape[1]
    own = own_count(lam, k)
    print(f"  polynomial class ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change over K {np.sqrt(cmu):.2e}")
    print("    lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:10]))
    print("    algebra Rayleigh quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(an.NAMES, rq)))
    print(f"    own count {own} (k = {k}); lambda{k + 1} / lambda{k} {lam[k] / max(lam[k - 1], 1e-300):.3g}")
    sa, sc, sq = scan_scaling(N, D)
    se, seq = scan_rotation(N, D)
    print(f"    pencil scans: scaling a {sa:.3f}, c {sc:.3f} (quotient {sq:.1e}; a + c {sa + sc:.3f}); stretched "
          f"rotation eps {se:.3f} (quotient {seq:.1e})", flush=True)
    probe = e45.Probe(dev)
    out.update(lam=lam[:16].tolist(), true_rq=rq, own=own, cmu=cmu, scan=dict(a=sa, c=sc, q=sq, eps=se, q_eps=seq))
    for name, m in (("span_k", k), ("span_own", own)):
        r = readouts(probe, C[:, :m], args.eps, args.a)
        out[name] = r
        print(f"    span of the {m} lowest ({name}): missing {r['missing']:.3f}; novel "
              + " ".join(f"{n} {v:.2f}" for n, v in zip(an.NAMES, r["novel"]))
              + f"; eps-hat {r['eps_hat']} (contained {r['rot_contained']:.3f}); a-hat {r['a_hat']}, c-hat "
              f"{r['c_hat']} (contained {r['scal_contained'][0]:.3f}, {r['scal_contained'][1]:.3f})", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", C=C[:, :16], lam=lam, N=N, D=D, u_train=dtr.u.numpy(), u_eval=dev.u.numpy(),
                 x=g.x.numpy(), dt=args.dt, eps=args.eps, a=args.a, nu=args.nu)
        print(f"  saved {stem}.json/.npz")


def check(args):
    g = an.Grid(args.n, args.L, args.score, eps=args.eps, a=args.a, nu=args.nu)
    t0 = time.time()
    u = an.trajectories(g, 4, 100, args.nt, args.dt, args.nsub)
    print(f"4 trajectories in {time.time() - t0:.1f} s; u in [{float(u.min()):.3f}, {float(u.max()):.3f}]; "
          f"RK4 |z|max nonlinear {args.dt / args.nsub * float(u.max()) * g.kmax2:.2f}, linear "
          f"{args.dt / args.nsub * float(g.frac.abs().max()):.2f} (stability 2.78)")
    mass = u.mean((-1, -2))
    print(f"mass drift {float((mass - mass[:, :1]).abs().max()):.1e}; rms of u - b0 within 1 of the seam / overall: "
          f"{float(((u - u[..., :1, :1])[..., :4, :] ** 2).mean().sqrt() / ((u - u[..., :1, :1]) ** 2).mean().sqrt()):.1e}")
    g2 = an.Grid(2 * args.n, args.L, args.score, eps=args.eps, a=args.a, nu=args.nu)
    uf = an.solve(g2, e45.initial(g2, 2, np.random.default_rng(100)), args.nt, args.dt, 2 * args.nsub)[:, -1, ::2, ::2]
    uc = an.solve(g, e45.initial(g, 2, np.random.default_rng(100)), args.nt, args.dt, args.nsub)[:, -1]
    print(f"against twice the grid and half the step (the same bumps): "
          f"{float((uf - uc).abs().max() / uc.abs().max()):.1e}")
    d = e45.Data(g, u, args.dt)
    op = an.ExactOp(g, args.dt, args.nsub)
    r, j = torch.tensor([0, 1]), torch.tensor([5, 20])
    q = e45.chars_poly(d, r, j, slice(None))[:, [1, 17, 40]]
    jv = op.transport(d.u[r, j], q, 3)[:, -1]
    eps_ = 1e-5 * float(d.u.abs().max()) / float(q.abs().max())
    upl = an.solve(g, (d.u[r, j][:, None] + eps_ * q).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    umi = an.solve(g, (d.u[r, j][:, None] - eps_ * q).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    fd = ((upl - umi) / (2 * eps_)).reshape(jv.shape)
    print(f"tangent vs central differences of the solver: rel err {float((jv - fd).norm() / fd.norm()):.1e}")


# ---------------------------------------------------------------------------------------------------------------------
# regression rows: weak SINDy in 2D with a local library and with nonlocal completions

def sym_iso(beta):
    return lambda KX, KY: -(KX ** 2 + KY ** 2) ** (beta / 2)


def sym_dir_x(beta):
    return lambda KX, KY: -np.abs(KX) ** beta


def sym_dir_y(beta):
    return lambda KX, KY: -np.abs(KY) ** beta


def sym_ell(eps, beta):
    return lambda KX, KY: -(KX ** 2 + eps ** 2 * KY ** 2) ** (beta / 2)


def sym_harm(beta, m):
    def f(KX, KY):
        K2 = KX ** 2 + KY ** 2
        c2 = np.where(K2 > 0, (KX ** 2 - KY ** 2) / np.where(K2 > 0, K2, 1.0), 0.0)
        return -K2 ** (beta / 2) * (1.0 if m == 0 else (c2 if m == 2 else 2 * c2 ** 2 - 1))
    return f


FAMILIES = {"iso": lambda prm: [("iso", sym_iso(prm[0]), prm[0])],
            "dir": lambda prm: [("dir_x", sym_dir_x(prm[0]), prm[0]), ("dir_y", sym_dir_y(prm[0]), prm[0])],
            "ell": lambda prm: [("ell", sym_ell(prm[0], prm[1]), prm[1])],
            "harm": lambda prm: [(f"harm{m}", sym_harm(prm[0], m), prm[0]) for m in (0, 2, 4)]}


class Lib2D:
    """The local columns once; multiplier columns per family and parameters; MSTLS through the normal equations for the
    profiles and the reference MSTLS for the reported model."""

    def __init__(self, U, x, y, t):
        self.w = mw2.WSINDy2D(POLYS, MAX_D)
        self.U = U
        self.b, self.Mlhs = self.w.prepare(U, x, y, t)
        self.local = [(j, p, q) for j, p, q in self.w.terms]
        self.G, self.M = self.w.columns(U, [(j, p, q, -1) for j, p, q in self.local])
        self.A, self.c, self.bb = self.G.T @ self.G, self.G.T @ self.b, float(self.b @ self.b)
        self.cache = {}

    def mult(self, fam, prm):
        key = (fam, tuple(np.round(prm, 6)))
        if key not in self.cache:
            specs = FAMILIES[fam](prm)
            G, M = self.w.columns(self.U, [(1, 0, 0, i) for i in range(len(specs))],
                                  tuple((f, order) for _, f, order in specs))
            self.cache[key] = dict(names=[n for n, _, _ in specs], G=G, M=M, X=self.G.T @ G, A=G.T @ G,
                                   c=G.T @ self.b)
        return self.cache[key]

    def resid(self, fam, prm):
        m = self.mult(fam, prm)
        A = np.block([[self.A, m["X"]], [m["X"].T, m["A"]]])
        c = np.concatenate([self.c, m["c"]])
        M = np.concatenate([self.M, m["M"]]) / self.Mlhs
        return e58.mstls_gram(A, c, self.bb, M, nterms=len(c))

    def fit(self, fam=None, prm=None):
        """The reference MSTLS on the local library plus the family's columns: (terms, residual, lambda)."""
        G, M, names = self.G, self.M, [("loc", j, p, q) for j, p, q in self.local]
        if fam is not None:
            m = self.mult(fam, prm)
            G, M = np.concatenate([G, m["G"]], 1), np.concatenate([M, m["M"]])
            names = names + [(n, 1, 0, 0) for n in m["names"]]
        W, lam, resid = self.w.mstls(G, self.b, M / self.Mlhs)
        terms = []
        for (kind, j, p, q), cc in zip(names, W):
            if cc != 0:
                terms.append(dict(kind=kind, j=j, p=p, q=q, c=float(cc), prm=None if kind == "loc" else
                                  [float(v) for v in prm]))
        return terms, float(resid), float(lam)

    def profile(self, fam, grids):
        """Parameters minimising the selected model's residual over the product of grids; the model there."""
        best = None
        for prm in np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T:
            r = self.resid(fam, prm)
            if best is None or r < best[0]:
                best = (r, prm)
        terms, resid, lam = self.fit(fam, best[1])
        return terms, resid, lam, [float(v) for v in best[1]]


def term_name(t):
    if t["kind"] == "loc":
        d = "".join(f"d{ax}^{n}" if n > 1 else f"d{ax}" for ax, n in (("x", t["p"]), ("y", t["q"])) if n)
        return f"{t['c']:+.4g} {d}(u^{t['j']})" if t["j"] else f"{t['c']:+.4g}"
    prm = ",".join(f"{v:.3g}" for v in t["prm"])
    return f"{t['c']:+.4g} {t['kind']}[{prm}] u"


def model_symbol(terms, KX, KY):
    """The linear part's symbol (local derivatives and multipliers) on a wavenumber grid."""
    s = np.zeros(KX.shape, dtype=complex)
    for t in terms:
        if t["kind"] == "loc" and t["j"] == 1:
            s += t["c"] * (1j * KX) ** t["p"] * (1j * KY) ** t["q"]
        elif t["kind"] != "loc":
            fam = {"iso": "iso", "dir_x": "dir", "dir_y": "dir", "ell": "ell", "harm0": "harm", "harm2": "harm",
                   "harm4": "harm"}[t["kind"]]
            spec = [sp for sp in FAMILIES[fam](t["prm"]) if sp[0] == t["kind"]][0]
            s += t["c"] * spec[1](KX, KY)
    return s


def model_algebra(terms, eps, a):
    """(stretched rotation: eps-hat or None; scaling: a-hat or None; true equation) of an identified model.
    Rotation: the nonlinear part is c1 d_xx(u^2) + c2 d_yy(u^2) (and u^j without derivatives), eps_n^2 = c2 / c1, and
    the linear symbol is a function of kx^2 + eps_n^2 ky^2 (tested on ellipses). Scaling: every term balances u_t at
    one (a, c) (exp56.balance with the multipliers' orders)."""
    nl = [t for t in terms if t["kind"] == "loc" and t["j"] >= 2]
    ok = all((t["p"], t["q"]) in ((2, 0), (0, 2), (0, 0)) and (t["j"] == 2 or (t["p"], t["q"]) == (0, 0))
             for t in nl)
    cxx = sum(t["c"] for t in nl if (t["p"], t["q"]) == (2, 0))
    cyy = sum(t["c"] for t in nl if (t["p"], t["q"]) == (0, 2))
    eps_hat = None
    if ok and cxx > 0 and cyy > 0:
        en = float(np.sqrt(cyy / cxx))
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        good = True
        for rad in (0.5, 1.0, 2.0, 4.0):
            KX, KY = rad * np.cos(th), rad * np.sin(th) / en
            v = model_symbol(terms, KX, KY)
            if np.abs(v).max() > 0 and (np.abs(v - v.mean()).max() > 1e-6 * np.abs(v).max()):
                good = False
        eps_hat = en if good else None
    orders = []
    for t in terms:
        if t["kind"] == "loc":
            orders.append((t["j"], t["p"] + t["q"]))
        else:
            orders.append((1, t["prm"][-1]))
    a_hat = e56.balance(orders) if orders else None
    kinds = sorted(t["kind"] for t in terms)
    true_eq = (kinds == ["ell", "loc", "loc"] and ok and len(nl) == 2 and cxx > 0 and cyy > 0
               and abs(np.sqrt(cyy / cxx) - eps) <= EPS_TOL
               and all(abs(t["prm"][0] - eps) <= EPS_TOL and abs(t["prm"][1] - a) <= DEG_TOL
                       for t in terms if t["kind"] == "ell"))
    return dict(eps_hat=eps_hat, a_hat=a_hat, true_eq=bool(true_eq),
                right_algebra=bool(eps_hat is not None and abs(eps_hat - eps) <= EPS_TOL and a_hat is not None
                                   and abs(a_hat - a) <= DEG_TOL))


def rhs_error(terms, g, U):
    """Relative error of the identified right-hand side against the true one on U [N, Nx, Ny, Nt] (spectral)."""
    KX, KY = np.meshgrid(2 * np.pi * np.fft.fftfreq(g.n, d=g.dx), 2 * np.pi * np.fft.fftfreq(g.n, d=g.dx),
                         indexing="ij")
    lapA = -(KX ** 2 + g.eps ** 2 * KY ** 2)
    frac = -g.nu * (KX ** 2 + g.eps ** 2 * KY ** 2) ** (g.a / 2)

    def op(sym, f):
        return np.real(np.fft.ifft2(sym[None, :, :, None] * np.fft.fft2(f, axes=(1, 2)), axes=(1, 2)))

    F0 = op(lapA, 0.5 * U ** 2) + op(frac, U)
    F = np.zeros_like(U)
    for t in terms:
        if t["kind"] == "loc":
            f = U ** t["j"] if t["j"] else np.ones_like(U)
            F += t["c"] * op((1j * KX) ** t["p"] * (1j * KY) ** t["q"], f)
        else:
            F += op(model_symbol([t], KX, KY), U)                  # the coefficient is inside the symbol
    return float(np.linalg.norm(F - F0) / np.linalg.norm(F0))


def regress(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    g = an.Grid(args.n, args.L, args.score, eps=args.eps, a=args.a, nu=args.nu)
    t0 = time.time()
    U = an.trajectories(g, args.n_train, dseed, args.nt, args.dt, args.nsub).numpy().transpose(0, 2, 3, 1)
    x = g.x.numpy()
    t = args.dt * np.arange(args.nt)
    lib = Lib2D(U, x, x, t)
    print(f"[aniso2d eps {args.eps:g} a {args.a:g}] seed {s} (data {dseed}): {len(U)} trajectories, {lib.w.info['rows']} "
          f"rows, test supports m {lib.w.info['mx']},{lib.w.info['my']},{lib.w.info['mt']}; {time.time() - t0:.0f} s",
          flush=True)
    rows = {}

    def put(name, terms, resid, lam, prm=None):
        alg = model_algebra(terms, args.eps, args.a)
        rows[name] = dict(terms=terms, names=" ".join(term_name(tt) for tt in terms), resid=resid, lam=lam, prm=prm,
                          eF=rhs_error(terms, g, U), **alg)
        r = rows[name]
        print(f"  {name:9s} {'TRUE ' if r['true_eq'] else 'wrong'} algebra {'right' if r['right_algebra'] else 'WRONG'}"
              f" (eps-hat {r['eps_hat']}, a-hat {r['a_hat']}) eF {r['eF']:.3g} resid {resid:.3g}"
              + (f" prm {prm}" if prm else "") + f"  u_t = {r['names']}", flush=True)

    t1 = time.time()
    put("ref", *lib.fit())
    print(f"  ({time.time() - t1:.0f} s)", flush=True)
    for name, fam, grids in (("iso-free", "iso", [BETAS]), ("dir-free", "dir", [BETAS]),
                             ("harm-free", "harm", [BETAS]), ("ell-free", "ell", [EPSS, BETAS])):
        t1 = time.time()
        terms, resid, lam, prm = lib.profile(fam, grids)
        put(name, terms, resid, lam, prm)
        print(f"  ({time.time() - t1:.0f} s)", flush=True)
    pick = "ref"
    for nxt in ("iso-free", "dir-free", "harm-free", "ell-free"):
        if rows[pick]["resid"] > NEST * rows[nxt]["resid"]:
            pick = nxt
    rows["nested"] = dict(rows[pick], picked=pick)
    print(f"  nested -> {pick}; total {time.time() - t0:.0f} s", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(dict(config=vars(args), data_seed=dseed, rows=rows,
                           wsindy={k: float(v) for k, v in lib.w.info.items()}), fh, indent=1)
        print(f"  saved {stem}.json")


def jobs(args):
    """(output, command) per exponent and seed: fins (exact) and regress on the seed's training data."""
    out = []
    for a in args.exponents.split(","):
        for s in args.seeds.split(","):
            stem = os.path.join(args.runs, f"fins_a{a}_s{s}")
            out.append((stem + ".json", ["exp60_aniso2d.py", "fins", "--a", a, "--seed", s, "--threads", "1",
                                         "--save", stem]))
            stem = os.path.join(args.runs, f"regress_a{a}_s{s}")
            out.append((stem + ".json", ["exp60_aniso2d.py", "regress", "--a", a, "--seed", s, "--threads", "1",
                                         "--save", stem]))
    return out


def summary(args):
    """results/aniso2d/summary.{json,md}: per exponent, FINS's readouts (exact; the FNO rows where they exist) and the
    regression rows' algebras, per seed."""
    res, lines = {}, ["# Anisotropic nonlocal dissipation in 2D (exp60): eps-hat, a-hat per row; counts over seeds\n"]
    rows_reg = ("ref", "iso-free", "dir-free", "harm-free", "ell-free", "nested")
    for a in args.exponents.split(","):
        cell = res.setdefault(a, {})
        for s in args.seeds.split(","):
            for tag, lab in (("fins", "FINS exact"), ("fins-fno", "FINS FNO2d")):
                p = os.path.join(args.runs, f"{tag}_a{a}_s{s}.json")
                if os.path.exists(p):
                    j = json.load(open(p))
                    cell.setdefault(lab, {})[s] = dict(scan=j["scan"], span_k=j["span_k"], span_own=j["span_own"],
                                                       own=j["own"], fno=j.get("fno_error_training"))
            p = os.path.join(args.runs, f"regress_a{a}_s{s}.json")
            if os.path.exists(p):
                j = json.load(open(p))
                for r in rows_reg:
                    if r in j["rows"]:
                        cell.setdefault(r, {})[s] = {key: j["rows"][r].get(key) for key in
                                                     ("eps_hat", "a_hat", "true_eq", "right_algebra", "prm", "names",
                                                      "eF", "resid", "picked")}
        al = float(a)
        lines.append(f"## a = {a} (eps = {an.EPS})\n")
        lines.append("| row | n | eps-hat within 0.01 (0.05) | a-hat within 0.02 (0.05, 0.1) | median eps-hat | median a-hat"
                     " | true equation |")
        lines.append("|---|---|---|---|---|---|---|")
        for lab, v in cell.items():
            if lab.startswith("FINS"):
                eh = [r["scan"]["eps"] for r in v.values()]
                ah = [r["scan"]["a"] for r in v.values()]
                te = "-"
            else:
                eh = [r["eps_hat"] for r in v.values()]
                ah = [r["a_hat"] for r in v.values()]
                te = f"{sum(bool(r['true_eq']) for r in v.values())}/{len(v)}"
            n = len(v)
            fe = lambda tol: sum(e is not None and abs(e - an.EPS) <= tol + 1e-9 for e in eh)     # verdicts' slack
            fa = lambda tol: sum(x is not None and abs(x - al) <= tol + 1e-9 for x in ah)
            me = np.median([e for e in eh if e is not None]) if any(e is not None for e in eh) else None
            ma = np.median([x for x in ah if x is not None]) if any(x is not None for x in ah) else None
            lines.append(f"| {lab} | {n} | {fe(0.01)} ({fe(0.05)}) | {fa(0.02)} ({fa(0.05)}, {fa(0.1)}) | "
                         f"{'none' if me is None else f'{me:.3f}'} | {'none' if ma is None else f'{ma:.3f}'} | {te} |")
        lines.append("")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "aniso2d")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["fins", "check", "reread", "regress", "run", "summary"])
    ap.add_argument("--op", default="exact", choices=["exact", "fno"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--fno-tag", default="")
    ap.add_argument("--train-fno-if-missing", action="store_true")
    ap.add_argument("--fno-n", type=int, default=512)
    ap.add_argument("--fno-width", type=int, default=32)
    ap.add_argument("--fno-modes", type=int, default=16)
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-epochs", type=int, default=60)
    ap.add_argument("--fno-batch", type=int, default=32)
    ap.add_argument("--fno-lr", type=float, default=1e-3)
    ap.add_argument("--exponents", default=",".join(f"{a:g}" for a in an.EXPONENTS))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "aniso2d_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--eps", type=float, default=an.EPS)
    ap.add_argument("--a", type=float, default=1.5)
    ap.add_argument("--nu", type=float, default=an.NU)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsub", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"fins": fins, "check": check, "reread": reread, "regress": regress, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
