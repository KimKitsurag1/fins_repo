"""Experiment 66 -- LieNLSD (Hu et al., ICML 2025) on exp60's anisotropic nonlocal dissipation (notebook 9;
README_EXPERIMENTS.md; a baseline that shares FINS's data without an operator).

exp47's LieNLSD-2D as published: an MLP F fitted to u_t on jets up to second order (central differences), F's Jacobian
in the determining system over FINS's 60 fields; trained on exp60's 16 trajectories per seed (data seed 4000 + s, a in
{1.25, 1.5, 1.75}), scored on its 8 held-out ones (5000 + s) against exp60's algebra. Readouts: the span of the 5
smallest right singular vectors and of LieNLSD's own count (below 0.01 of the largest); with N = W'W and
D = diag(sc^2), LieNLSD's column scaling, exp60's rotation and scaling scans and exp60_readouts' sl(2). Variants:
fitted, and the MLP at initialisation (untrained).

Run:  python exp66_lienlsd_aniso.py --a 1.5 --seed 42 --save <stem>
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import aniso2d as an                                       # noqa: E402
import exp45_neural_2d as e45                              # noqa: E402
import exp47_lienlsd_2d as e47                             # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp60_aniso2d as e60                                # noqa: E402
import exp60_readouts as e60r                              # noqa: E402
from method_lienlsd import Adan, RHSNet                    # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def fit(S, sl, epochs, seed, n_fit, n_val, device, batch=256, lr=1e-3, every=10):
    """exp47.fit on a device: MLP 200 x 3 sigmoid, Adan, MSE, batch 256, the checkpoint (every 10 epochs) with the
    smallest validation error on the last two training trajectories."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    B = S["U"].shape[0]
    tr = e47.block_points(S, sl, rng, n_fit, np.arange(B - 2))
    va = e47.block_points(S, sl, rng, n_val, np.arange(B - 2, B))
    Xtr, ytr, Xva, yva = e47.gather(S, tr, e47.COLS), S["UT"][tr], e47.gather(S, va, e47.COLS), S["UT"][va]
    net = RHSNet(Xtr.mean(0), Xtr.std(0) + 1e-12, ytr.mean(), ytr.std() + 1e-12).to(device)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32, device=device), torch.tensor(ytr, dtype=torch.float32,
                                                                                device=device)[:, None]
    Xv, yv = torch.tensor(Xva, dtype=torch.float32, device=device), torch.tensor(yva, dtype=torch.float32,
                                                                                device=device)[:, None]
    opt = Adan(net.parameters(), lr) if Adan is not None else torch.optim.Adam(net.parameters(), lr)

    def val():
        with torch.no_grad():
            return float(((net(Xv) - yv) ** 2).mean() / (yv ** 2).mean())

    log = dict(optimizer=type(opt).__name__, epochs=epochs, val=[(0, val())])
    best, state = log["val"][0][1], copy.deepcopy(net.state_dict())
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for b in range(0, len(Xt), batch):
            i = perm[b:b + batch]
            loss = tf.mse_loss(net(Xt[i]), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % every == 0 or ep + 1 == epochs:
            v = val()
            log["val"].append((ep + 1, v))
            if v < best:
                best, state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.eval()
    log["val_rel_mse"] = best
    return net.cpu(), log


def run(args):
    s, a = args.seed, args.a
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = an.Grid(args.n, args.L, args.score, eps=an.EPS, a=a, nu=an.NU)
    t0 = time.time()
    dtr = e45.Data(g, an.trajectories(g, args.n_train, 4000 + s, args.nt, args.dt, args.nsub), args.dt)
    dev = e45.Data(g, an.trajectories(g, args.n_eval, 5000 + s, args.nt, args.dt, args.nsub), args.dt, dtr.s)
    probe = e45.Probe(dev)
    Ct = an.true_coefficients(an.EPS, a)
    true_q = probe.poly(Ct)[1]
    sc = dtr.s
    tgrid = dtr.t.numpy()[2:-2] / sc["T"]
    S = e47.scaled(e47.jets(dtr.u.numpy(), g.dx, args.dt, "fd"), sc)
    S["Xg"], S["Yg"] = (g.X / sc["Lx"]).numpy(), (g.Y / sc["Lx"]).numpy()
    Se = e47.scaled(e47.jets(dev.u.numpy(), g.dx, args.dt, "fd"), sc)
    Pe = np.stack([Se[k][:, :, g.sl, g.sl].ravel() for k in e47.COLS], -1)
    ute = Se["UT"][:, :, g.sl, g.sl].ravel()
    rng = np.random.default_rng(0)
    pts = e47.block_points(S, g.sl, rng, args.n_points * args.n_traj, np.arange(args.n_traj))
    P = e47.gather(S, pts, e47.COLS)
    fns = e47.library()
    print(f"[aniso2d a {a:g}, LieNLSD] seed {s}: {args.n_train} trajectories (data seed {4000 + s}), {len(P)} "
          f"determining points; {time.time() - t0:.0f} s; {device}", flush=True)
    out = dict(config=vars(args))
    for variant in ("fitted", "untrained"):
        t0 = time.time()
        net, log = fit(S, g.sl, args.epochs if variant == "fitted" else 0, s, args.n_fit, args.n_val, device)
        J = e47.net_jacobian(net, P)
        with torch.no_grad():
            fe = net(torch.tensor(Pe, dtype=torch.float32))[:, 0].numpy()
        rhs = float(np.linalg.norm(fe - ute) / np.linalg.norm(ute))
        W = e47.determining(S, pts, J, fns, tgrid)
        C, n_own, sv = e47.readout(W, 5)
        scl = np.sqrt((W ** 2).mean(0))
        scl = np.where(scl > 1e-12, scl, 1.0)
        N, D = W.T @ W, np.diag(scl ** 2)
        sa, sc_, sq = e60.scan_scaling(N, D)
        se, seq = e60.scan_rotation(N, D)
        rq = [float(c @ N @ c / (c @ D @ c)) for c in Ct.T]
        res = dict(missing_5=e45.missing(probe.poly(C)[1], true_q), n_own=n_own,
                   missing_own=e45.missing(probe.poly(C[:, :n_own])[1], true_q), sv_rel=sv, rhs_rel=rhs,
                   val_rel_mse=log["val_rel_mse"], true_rq=rq, scan=dict(a=sa, c=sc_, q=sq, eps=se, q_eps=seq),
                   sl2=e60r.sl2(N, D), time=time.time() - t0)
        out[variant] = res
        print(f"  {variant:9s} missing (5) {res['missing_5']:.3f}; own count {n_own}: missing {res['missing_own']:.3f}; "
              f"F rel. error {rhs:.4f} (val {log['val_rel_mse']:.2e}); scans: eps {se:.3f} ({seq:.1e}), a {sa:.3f} "
              f"c {sc_:.3f} ({sq:.1e}); sl(2) eps {res['sl2']['eps']}; sv " + " ".join(f"{x:.1e}" for x in sv[:7])
              + f" ({res['time']:.0f} s)", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"  saved {stem}.json")


def jobs(args):
    out = []
    for a in ("1.25", "1.5", "1.75"):
        for s in args.seeds.split(","):
            stem = os.path.join(args.runs, f"lienlsd_a{a}_s{s}")
            out.append((stem + ".json", ["exp66_lienlsd_aniso.py", "one", "--a", a, "--seed", s, "--save", stem]))
    return out


def summary(args):
    res, lines = {}, ["# LieNLSD on exp60 (exp66)\n", "| a | variant | n | missing (5), median | a-hat (scan) | "
                                                           "eps-hat (scan) | sl(2) eps |", "|---|---|---|---|---|---|---|"]
    for a in ("1.25", "1.5", "1.75"):
        for s in args.seeds.split(","):
            p = os.path.join(args.runs, f"lienlsd_a{a}_s{s}.json")
            if os.path.exists(p):
                res.setdefault(a, {})[s] = json.load(open(p))
        for v in ("fitted", "untrained"):
            cell = [r[v] for r in res.get(a, {}).values() if v in r]
            if not cell:
                continue
            rng = (lambda xs, f="{:.3f}": f"{f.format(min(xs))} to {f.format(max(xs))}" if xs else "-")
            lines.append(f"| {a} | {v} | {len(cell)} | {np.median([c['missing_5'] for c in cell]):.3f} | "
                         f"{rng([c['scan']['a'] for c in cell])} | {rng([c['scan']['eps'] for c in cell])} | "
                         f"{rng([c['sl2']['eps'] for c in cell if c['sl2']['eps'] is not None])} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "aniso2d")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "lienlsd_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "lienlsd_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["one", "run", "summary"])
    ap.add_argument("--a", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "lienlsd_aniso_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-fit", type=int, default=2 ** 17)
    ap.add_argument("--n-val", type=int, default=2 ** 15)
    ap.add_argument("--n-traj", type=int, default=10)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsub", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    if args.what == "run":
        e51.run(args, jobs(args))
    elif args.what == "summary":
        summary(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
