"""Experiment 68 -- an emulator audit with a known answer (notebook 10; README_EXPERIMENTS.md).

FNO2d emulators of pyqg's flow at 128^2 with PG8's two limits removed (40 Fourier modes, 1024 training trajectories):
f-plane plain, rot (every training triple rotated about the box centre by a uniform random angle: a quarter turn, then
three Fourier shears, exact for band-limited periodic fields) and grid (the coordinates as two input channels, the FNO
paper's default, which breaks the translations); beta-plane plain, whose physics breaks the rotation. Then:
  train   -- exp61's FNO2d (residual step, one-step plus two-step loss, Adam, cosine) on pyqg's trajectories of data seed
             6000 + s, a fixed number of steps over random pairs; resumable from <checkpoint>.partial.
  error   -- one-step and K-step error against pyqg's data relative to the increment on the scored block (exp45), on
             the audit's trajectories (data seeds 4000 + s and 5000 + s).
  commute -- the direct reference: commutators ||F(T q) - T F(q)|| / ||F(q) - q|| over held-out pyqg states (snapshots
             10, 20, ..., 50 of data seed 5000 + s) for rotations by 90, 30 and 2 degrees and shifts by 32, 8 and 0.5
             cells along x and y, the port's (qg2d) alongside.
  fins    -- exp65_emulator_audit.py fins with --fno-tag: FINS through the emulator on its own rollouts.

Run:  python exp68_emulator_pairs.py train --arm rot --seed 42
      python exp68_emulator_pairs.py commute --arm rot --seed 42 --save <stem>
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pyqg
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
ARMS = ("plain", "rot", "grid")
ANGLES = (90.0, 30.0, 2.0)
SHIFTS = (32.0, 8.0, 0.5)


def tag_of(args):
    return f"_ex68{args.arm}{args.tag_suffix}"


def grid_of(args):
    return qg2d.regime_grid(args.regime, args.n, args.L, args.score)


def pyqg_trajectories32(g, n_traj, seed, nt, dt, nsub):
    """exp61.pyqg_trajectories in float32, one trajectory at a time (the same initial states)."""
    q0 = qg2d.initial(g, n_traj, np.random.default_rng(seed))
    out = np.zeros((n_traj, nt, g.n, g.n), dtype=np.float32)
    for r in range(n_traj):
        m = pyqg.BTModel(nx=g.n, L=g.L, beta=g.beta, rd=g.rd, rek=g.rek, U=g.U, filterfac=g.filterfac, dt=dt / nsub,
                         tmax=dt * nt, twrite=10 ** 9, tavestart=10 * dt * nt, log_level=0)
        m.set_q(q0[r].double().numpy().T[None].copy())
        out[r, 0] = np.asarray(m.q)[0].T
        for j in range(1, nt):
            for _ in range(nsub):
                m._step_forward()
            out[r, j] = np.asarray(m.q)[0].T
    return torch.from_numpy(out)


def shear(q, a, dim, dx):
    """q(r - a * c e_dim) for the other coordinate c, by phases along dim: a [B] (q [B, n, n])."""
    n = q.shape[-1]
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=q.dtype, device=q.device)
    k[n // 2] = 0.0
    c = (torch.arange(n, dtype=q.dtype, device=q.device) - n // 2) * dx
    if dim == -2:
        ph = torch.exp(-1j * k[None, :, None] * a[:, None, None] * c[None, None, :])
    else:
        ph = torch.exp(-1j * k[None, None, :] * a[:, None, None] * c[None, :, None])
    h = torch.fft.fft(q.to(torch.complex128 if q.dtype == torch.float64 else torch.complex64), dim=dim)
    return torch.fft.ifft(h * ph, dim=dim).real


def quarter(q, k):
    """Rotation by k quarter turns about the grid point n / 2 (exact): q'[i, j] = q[j, -i] per turn."""
    n = q.shape[-1]
    idx = (-torch.arange(n, device=q.device)) % n
    for _ in range(int(k) % 4):
        q = q.transpose(-1, -2)[..., idx, :]
    return q


def rotate(q, theta, dx):
    """Counter-clockwise rotation of fields q [B, n, n] about the box centre by angles theta [B] (radians)."""
    theta = torch.as_tensor(theta, dtype=q.dtype, device=q.device).reshape(-1)
    k = torch.round(theta / (math.pi / 2)).long()
    phi = theta - k.to(q.dtype) * (math.pi / 2)
    out = q.clone()
    for kk in range(4):
        sel = (k % 4) == kk
        if bool(sel.any()):
            out[sel] = quarter(q[sel], kk)
    a, b = -torch.tan(phi / 2), torch.sin(phi)
    out = shear(out, a, -2, dx)
    out = shear(out, b, -1, dx)
    return shear(out, a, -2, dx)


def load_data(args, g, n_traj, seed):
    """pyqg's trajectories [n_traj, nt, n, n] in float32, cached as .npy under --cache."""
    path = os.path.join(args.cache, f"pyqg_{args.regime}_n{args.n}_traj{n_traj}_seed{seed}.npy")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path))
    os.makedirs(args.cache, exist_ok=True)
    t0 = time.time()
    u = pyqg_trajectories32(g, n_traj, seed, args.nt, args.dt, args.nsub)
    np.save(path + ".tmp.npy", u.numpy())
    os.replace(path + ".tmp.npy", path)
    print(f"  generated {n_traj} pyqg trajectories (seed {seed}) in {time.time() - t0:.0f} s -> {path}", flush=True)
    return u


def read_partial(part, dev):
    """A resumable checkpoint, or None (an unreadable one, e.g. truncated, is removed)."""
    if not os.path.exists(part):
        return None
    try:
        return torch.load(part, map_location=dev, weights_only=False)
    except (EOFError, RuntimeError, OSError, pickle.UnpicklingError) as err:
        print(f"unreadable {part} ({type(err).__name__}: {err}); removed, starting over", flush=True)
        os.remove(part)
        return None


def train(args):
    g = grid_of(args)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.var, args.fno_tag = "q", tag_of(args)
    path = e61.fno_path(args)
    if os.path.exists(path):
        print(f"exists: {path}")
        return
    u = load_data(args, g, args.n_traj, 6000 + args.seed)
    torch.manual_seed(args.seed)
    model = e45.FNO2d(args.width, args.modes, args.layers, grid=args.arm == "grid")
    inc = float((u[:, 1:] - u[:, :-1]).std())
    model.norm.copy_(torch.tensor([float(u.mean()), float(u.std()), inc]))
    model = model.to(dev)
    ud = u.to(dev)
    B, T = ud.shape[:2]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + args.seed)
    start, log = 0, []
    part = path + ".partial"
    ck = read_partial(part, dev)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        start, log = ck["step"], ck["log"]
        print(f"resumed from {part} at step {start}", flush=True)
    print(f"training FNO2d ({args.regime}, {args.arm}) on {B} trajectories ({B * (T - 2)} triples), width {args.width}, modes "
          f"{args.modes}, {args.layers} layers, {args.steps} steps of {args.batch}, lr {args.lr}, {dev}", flush=True)
    t0, tot, cnt = time.time(), 0.0, 0
    for st in range(start, args.steps):
        r = torch.randint(0, B, (args.batch,), device=dev, generator=gen)
        j = torch.randint(0, T - 2, (args.batch,), device=dev, generator=gen)
        x0, x1, x2 = ud[r, j], ud[r, j + 1], ud[r, j + 2]
        if args.arm == "rot":
            th = 2 * math.pi * torch.rand(args.batch, device=dev, generator=gen)
            x0, x1, x2 = (rotate(x, th, g.dx) for x in (x0, x1, x2))
        p1 = model(x0)
        p2 = model(p1)
        loss = (((p1 - x1) ** 2).mean() + ((p2 - x2) ** 2).mean()) / inc ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot, cnt = tot + loss.item(), cnt + 1
        if (st + 1) % args.log_every == 0 or st + 1 == args.steps:
            log.append((st + 1, tot / cnt))
            print(f"  step {st + 1}: loss {tot / cnt:.3e} ({time.time() - t0:.0f} s)", flush=True)
            tot, cnt = 0.0, 0
        if (st + 1) % args.ckpt_every == 0 and st + 1 < args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=gen.get_state(), step=st + 1, log=log), part + ".tmp")
            os.replace(part + ".tmp", part)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    cfg = dict(width=args.width, modes=args.modes, layers=args.layers)
    if args.arm == "grid":
        cfg["grid"] = True
    torch.save(dict(state=sd, cfg=cfg, log=log,
                    arm=args.arm, steps=args.steps, n_traj=B), path)
    if os.path.exists(part):
        os.remove(part)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


def model_of(args, g):
    args.var, args.fno_tag, args.train_fno_if_missing = "q", tag_of(args), False
    return e61.load_fno(args, g.cast(torch.float32)).double()


def error(args):
    g = grid_of(args)
    model = model_of(args, g)
    out = dict(config=vars(args))
    for name, seed, nn_ in (("training", 4000 + args.seed, args.n_train), ("held-out", 5000 + args.seed, args.n_eval)):
        d = e45.Data(g, e61.pyqg_trajectories(g, nn_, seed, args.nt, args.dt, args.nsub), args.dt)
        e1, eK, per = e45.operator_errors(model, d, args.K)
        out[name] = dict(one=e1, K=eK, per=per.tolist())
        print(f"  {args.regime} {args.arm} s{args.seed} ({name}, data seed {seed}): one step {e1:.4f} (per trajectory {per.min():.4f}-"
              f"{per.max():.4f}), {args.K} steps {eK:.4f}", flush=True)
    save(args, out)


def snapshot_map(g, q, dt, nsub):
    for _ in range(nsub):
        q = qg2d.step(g, q, dt / nsub)
    return q


def translate(q, cells, dim):
    """Shift by a number of cells along dim (spectral, exact for band-limited periodic fields)."""
    n = q.shape[-1]
    if float(cells).is_integer():
        return torch.roll(q, int(cells), dim)
    k = torch.fft.fftfreq(n, dtype=q.dtype, device=q.device) * n
    k[n // 2] = 0.0
    ph = torch.exp(-2j * math.pi * k * cells / n)
    ph = ph[:, None] if dim == -2 else ph[None, :]
    return torch.fft.ifft(torch.fft.fft(q, dim=dim) * ph, dim=dim).real


def commutator(F, X, transform):
    with torch.no_grad():
        FX, FTX = F(X), F(transform(X))
    num = ((FTX - transform(FX)) ** 2).sum((-2, -1))
    den = ((FX - X) ** 2).sum((-2, -1))
    per = (num / den).sqrt()
    return float((num.sum() / den.sum()).sqrt()), float(per.median())


def transforms(g):
    out = {}
    for ang in ANGLES:
        out[f"rotation {ang:g}"] = (lambda a: lambda q: rotate(q, torch.full((q.shape[0],), math.radians(a),
                                                                             dtype=q.dtype), g.dx))(ang)
    for c in SHIFTS:
        out[f"d_x {c:g}"] = (lambda c_: lambda q: translate(q, c_, -2))(c)
        out[f"d_y {c:g}"] = (lambda c_: lambda q: translate(q, c_, -1))(c)
    return out


def commute(args):
    g = grid_of(args)
    qev = e61.pyqg_trajectories(g, args.n_eval, 5000 + args.seed, args.nt, args.dt, args.nsub)
    X = qev[:, list(range(args.snap_first, args.nt, args.snap_every))].reshape(-1, g.n, g.n)
    ops = {"port": lambda q: snapshot_map(g, q, args.dt, args.nsub)}
    if args.arm != "port":
        model = model_of(args, g)
        ops["emulator"] = model
    out = dict(config=vars(args), n_states=int(X.shape[0]))
    tr = transforms(g)
    for name, F in ops.items():
        res = {}
        for key, T in tr.items():
            agg, med = commutator(F, X, T)
            res[key] = dict(rms=agg, median=med)
        out[name] = res
        print(f"  {args.regime} {args.arm} s{args.seed} {name}: commutator / increment " + "  ".join(
            f"{k} {v['rms']:.2e}" for k, v in res.items()), flush=True)
    save(args, out)


def save(args, out):
    if not args.save:
        return
    stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"  saved {stem}.json")


ROWS = (("f", "plain"), ("f", "grid"), ("f", "rot"), ("beta", "plain"))


def stem_of(runs, what, regime, arm, s):
    return os.path.join(runs, f"ex68_{what}_{regime}_{arm}_s{s}")


def jobs(args):
    out = []
    for s in args.seeds.split(","):
        for regime, arm in ROWS:
            base = ["--regime", regime, "--arm", arm, "--seed", s]
            for what in ("error", "commute"):
                stem = stem_of(args.runs, what, regime, arm, s)
                out.append((stem + ".json", ["exp68_emulator_pairs.py", what] + base + ["--save", stem]))
            stem = stem_of(args.runs, "fins", regime, arm, s)
            out.append((stem + ".json", ["exp65_emulator_audit.py", "fins", "--regime", regime, "--seed", s,
                                         "--fno-tag", f"_ex68{arm}", "--save", stem]))
    return out


GENS = ("d_x", "d_y", "d_t", "rotation", "D_x", "D_t", "D_beta")


def summary(args):
    res, lines = {}, ["# Emulator pairs with and without rotation augmentation (exp68)\n",
                      "| regime | seed | arm | held-out error (1 step) | commutator: rotation 90 / 30 / 2 deg, d_x 8 "
                      "/ 0.5 cells | rotation plane "
                      "(FINS) | quotients (FINS) | span of k, missing |", "|---|---|---|---|---|---|---|---|"]
    for s in args.seeds.split(","):
        for regime, arm in ROWS:
            cell = {}
            for what in ("error", "commute", "fins"):
                p = stem_of(args.runs, what, regime, arm, s) + ".json"
                if os.path.exists(p):
                    cell[what] = json.load(open(p))
            if not cell:
                continue
            res.setdefault(regime, {}).setdefault(arm, {})[s] = {k: {kk: vv for kk, vv in v.items() if kk != "config"}
                                                                 for k, v in cell.items()}
            e = cell.get("error", {}).get("held-out", {}).get("one")
            c = cell.get("commute", {}).get("emulator", {})
            f = cell.get("fins", {})
            lines.append(f"| {regime} | {s} | {arm} | " + (f"{e:.4f} | " if e is not None else "- | "))
            lines[-1] += (" / ".join(f"{c[k]['rms']:.1e}" for k in ("rotation 90", "rotation 30", "rotation 2",
                                                                    "d_x 8", "d_x 0.5")) if c else "-") + " | "
            lines[-1] += (f"{f['planes']['rot_lam'][0]:.1e} | " + " ".join(f"{gn} {f['quotients'][gn]:.1e}" for gn in GENS)
                          + f" | {f['span_k']['missing']:.3f} |") if f else "- | - | - |"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pyqg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pairs_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "pairs_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check(args):
    """Rotation accuracy on an analytic blob and on pyqg states; the port's commutator."""
    g = grid_of(args)
    X, Y = g.X, g.Y
    blob = lambda x0, y0: torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / 0.6 ** 2)[None]
    for ang in (30.0, 90.0, 120.0, -100.0, 2.0):
        t = math.radians(ang)
        x0, y0 = 2.0, 1.0
        want = blob(x0 * math.cos(t) - y0 * math.sin(t), x0 * math.sin(t) + y0 * math.cos(t))
        got = rotate(blob(x0, y0), torch.tensor([t], dtype=X.dtype), g.dx)
        print(f"  blob rotated by {ang:g} deg: relative error {float((got - want).norm() / want.norm()):.1e}")
    q = pyqg_trajectories32(g, 2, 5007, 11, args.dt, args.nsub)[:, -1].double()
    th = torch.tensor([0.7, -2.1], dtype=q.dtype)
    back = rotate(rotate(q, th, g.dx), -th, g.dx)
    print(f"  pyqg states rotated and back: relative error {float((back - q).norm() / q.norm()):.1e}")
    F = lambda z: snapshot_map(g, z, args.dt, args.nsub)
    for key, T in transforms(g).items():
        agg, _ = commutator(F, q, T)
        print(f"  port commutator, {key}: {agg:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["train", "error", "commute", "run", "summary", "check"])
    ap.add_argument("--arm", default="plain", choices=list(ARMS) + ["port"])
    ap.add_argument("--regime", default="f", choices=["f", "beta"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(RESULTS, "pairs_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=os.path.join(RESULTS, "pairs_cache"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--n-traj", type=int, default=1024)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=40)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--nsub", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--snap-first", type=int, default=10)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"train": train, "error": error, "commute": commute, "summary": summary, "check": check}[args.what](args)


if __name__ == "__main__":
    main()
