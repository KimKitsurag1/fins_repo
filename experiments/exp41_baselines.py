"""Experiment 41 -- baselines on the protocol data (PROTOCOL.md, section 6).

Each method is trained on the protocol's training trajectories of seed s; its generators are
evaluated as (xi, eta, phi) fields on the ten evaluation trajectories of data seed 1000 + s, the
same trajectories FINS is scored on. exp42 scores every method on one region with one metric.
Equation-aware methods get the equation the data satisfy (LIG: cfg.lig_eq; LieNLSD: run_lienlsd_general, library of
the protocol degree). LieNLSD_F: LieNLSD as published (F fitted to u_t on finite-difference jets; --untrained: F at
initialisation); LieNLSD_RHS: the same with the exact right-hand side; *_spec: both on FINS's derivatives (spectral
in x, fourth order in t). Each saves k generators (ascending singular value) and the method's own count (nown_*).
A suffix @eval builds LieNLSD's determining system on the evaluation trajectories, the ones FINS's pencil uses, instead
of the training ones (exp48). --noise / --thin (exp48): training and evaluation data corrupted as in corrupt.py; the
fields are evaluated on the clean evaluation data, which is what the file stores as u (the other baselines train on
the corrupted data and extract their generators on the clean evaluation data).

Run:  python exp41_baselines.py --pde heat --seed 42 --out baselines_heat_s42.npz
      python exp41_baselines.py --pde heat --seed 42 --untrained --out untrained_heat_s42.npz
"""

import argparse
import json
import os
import sys
import time
import traceback

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp40_window as e40                                 # noqa: E402
import corrupt as cr                                       # noqa: E402
import bo                                                  # noqa: E402
import ks                                                  # noqa: E402
from method_augerino import extract_augerino_generators, train_augerino  # noqa: E402
from method_laligan import extract_laligan_generators, train_laligan     # noqa: E402
from method_liegan import extract_liegan_generators, train_liegan        # noqa: E402
from method_lienlsd import FIT_EPOCHS, lienlsd_fields, run_lienlsd_general, run_lienlsd_published  # noqa: E402
from method_lig import extract_lig_generators, train_lig                 # noqa: E402

DEGREE = {"burgers": 2, "kdv": 2, "heat": 3, "bo": 2, "ks": 2}
K_CLASS = {"burgers": 5, "kdv": 4, "heat": 10, "bo": 4, "ks": 3}
METHODS = ("LieNLSD", "LieGAN", "LaLiGAN", "Augerino", "LIG")      # LIG last: the most memory-hungry
# also available: LIG_k (LIG with n_sym = class dimension instead of Ko et al.'s cap of 4); LieNLSD_F, LieNLSD_RHS


def equation(pde, cfg):
    """Coefficients of the equation the protocol data satisfy (ground_truth.py, data_generation.py)."""
    if pde == "heat":
        return {"alpha": cfg.alpha_heat}
    if pde == "burgers":
        return {"a": 2 * np.pi, "b": 4 * np.pi ** 2 * cfg.nu_burgers}
    if pde == "bo":
        return {}                                   # nonlocal: no jet-space right-hand side
    if pde == "ks":
        return {}                                   # u_t = -u u_x - u_xx - u_xxxx (method_lienlsd.exact_rhs)
    return {"a": 6.0, "c": 1.0}


def protocol_data(pde, cfg, seed, dseed, n_train, n_eval):
    """Training tensor [N, 1, Nx, Nt] with its grid, and the evaluation u, xs, ts, built exactly as
    exp40 (Burgers, heat) and exp39 (KdV) build them."""
    if pde in ("kdv", "bo", "ks"):
        T, Nx, Nt = {"bo": bo.T, "ks": ks.T}.get(pde, 0.05), 128, 128
        _, x, t = e39.grid(pde, cfg, Nx, Nt, T)
        cache = os.environ.get("FINS_DATA_CACHE")       # optional: the 1024 KdV training solves take ~1.5 min
        path = os.path.join(cache, f"{pde}_mixed_train_s{seed}_n{n_train}.npy") if cache else None
        if path and os.path.exists(path):
            tr = np.load(path)
        else:
            tr = e39.training_data(pde, n_train, seed, "mixed", x, T, Nt, 0.002).astype(np.float32)
            if path:
                os.makedirs(cache, exist_ok=True)
                np.save(path + f".{os.getpid()}.npy", tr)
                os.replace(path + f".{os.getpid()}.npy", path)
        u, _, ts = e39.evaluation_data(pde, "periodic", n_eval, dseed, x, t, 0.002)
        return torch.tensor(tr, dtype=torch.float32)[:, None], x, t, u, np.tile(x, (n_eval, 1)), ts
    smooth, only_L = (True, 1.0) if pde == "heat" else (False, None)
    xs_tr, ts_tr, tr, _ = e40.trajectories(pde, cfg, n_train, seed, smooth, only_L)
    xs, ts, u, _ = e40.trajectories(pde, cfg, n_eval, dseed, smooth, only_L)
    return torch.tensor(tr, dtype=torch.float32)[:, None], xs_tr[0], ts_tr[0], u, xs, ts


def as_fields(gens):
    return np.stack([np.stack([np.asarray(c, dtype=np.float64) for c in g]) for g in gens])


def run_method(name, pde, tr, x, t, u, xs, ts, cfg, dev, extra, u_seen=None):
    """Fields [n_eval, k, 3, Nx, Nt] of the method's generators on each evaluation trajectory u (clean). u_seen: the
    evaluation trajectories as the methods see them (corrupted; default u)."""
    u_seen = u if u_seen is None else u_seen
    ev = torch.tensor(u, dtype=torch.float32)[:, None]
    n = len(u)
    if name == "LieGAN":
        g = train_liegan(tr, pde, x, t, cfg, dev)
        return np.stack([as_fields(extract_liegan_generators(g, ev[r:r + 1], xs[r], ts[r], cfg, dev)[0])
                         for r in range(n)])
    if name == "LaLiGAN":
        g, enc, dec = train_laligan(tr, cfg, dev)
        return np.stack([as_fields(extract_laligan_generators(g, enc, dec, ev[r:r + 1], cfg, dev)[0])
                         for r in range(n)])
    if name in ("LIG", "LIG_k"):
        # LIG: Ko et al.'s cap of 4 generators; LIG_k: n_sym = class dimension (PROTOCOL.md, section 8)
        cfg.lig_max_sym = K_CLASS[pde] if name == "LIG_k" else 4
        g = train_lig(tr, pde, x, t, cfg, dev)
        return np.stack([as_fields(extract_lig_generators(g, ev[r:r + 1], xs[r], ts[r], cfg, dev)[0])
                         for r in range(n)])
    if name == "Augerino":
        aug, model = train_augerino(tr, pde, x, t, cfg, dev)
        return np.stack([as_fields(extract_augerino_generators(aug, model, ev[r:r + 1], xs[r], ts[r], cfg, dev)[0])
                         for r in range(n)])
    base, _, where = name.partition("@")
    if base in ("LieNLSD_F", "LieNLSD_RHS", "LieNLSD_F_spec", "LieNLSD_RHS_spec"):
        epochs = None if base.startswith("LieNLSD_RHS") else cfg.lienlsd_fit_epochs
        jets = "spectral" if base.endswith("_spec") else "fd"
        system = (u_seen, ts[0]) if where == "eval" else None
        coeffs, n_own, exps, _, diag = run_lienlsd_published(tr, pde, x, t, DEGREE[pde], K_CLASS[pde],
                                                             equation(pde, cfg), epochs, u_seen,
                                                             seed=cfg.lienlsd_seed, jets=jets,
                                                             n_fit=cfg.lienlsd_fit_points,
                                                             n_val=cfg.lienlsd_val_points, system=system)
        extra[name] = dict(n_own=n_own, diag=diag)
        return np.stack([as_fields(lienlsd_fields(coeffs, exps, xs[r], ts[r], u[r])) for r in range(n)])
    coeffs, exps, _ = run_lienlsd_general(tr, pde, x, t, cfg, dev, DEGREE[pde], equation(pde, cfg),
                                          n_max=K_CLASS[pde])
    return np.stack([as_fields(lienlsd_fields(coeffs, exps, xs[r], ts[r], u[r])) for r in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["burgers", "heat", "kdv", "bo", "ks"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None, help="default 1000 + seed")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiplies every method's epoch count (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None, help="cfg.batch_size (LIG uses min(it, 4))")
    ap.add_argument("--lig-epochs", type=int, default=None,
                    help="LIG epoch cap (its schedule scales with it: lr drop at half, Sobolev term in the last 20 percent)")
    ap.add_argument("--untrained", action="store_true",
                    help="zero epochs: each method's generators at initialisation (own-class null; LieNLSD has no training)")
    ap.add_argument("--lienlsd-fit-points", type=int, default=None,
                    help="training points of LieNLSD's F (default 10 Nx Nt; the official ten ICs hold ~1e6)")
    ap.add_argument("--lienlsd-val-points", type=int, default=None, help="validation points of F (default 2 Nx Nt)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise, times each trajectory's std (corrupt.py)")
    ap.add_argument("--thin", type=int, default=1, help="keep every thin-th snapshot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    cfg.n_sym = K_CLASS[args.pde]
    cfg.lig_eq = equation(args.pde, cfg)
    for f in ("epochs_liegan", "epochs_laligan", "epochs_lig", "epochs_augerino", "lienlsd_epochs"):
        setattr(cfg, f, max(1, int(round(getattr(cfg, f) * args.epochs_scale))))
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lig_epochs:
        cfg.epochs_lig = args.lig_epochs
    cfg.lienlsd_fit_epochs = max(1, int(round(FIT_EPOCHS[args.pde] * args.epochs_scale)))
    cfg.lienlsd_seed = args.seed
    cfg.lienlsd_fit_points, cfg.lienlsd_val_points = args.lienlsd_fit_points, args.lienlsd_val_points
    if args.untrained:
        for f in ("epochs_liegan", "epochs_laligan", "epochs_lig", "epochs_augerino", "lienlsd_fit_epochs"):
            setattr(cfg, f, 0)
    dseed = 1000 + args.seed if args.data_seed is None else args.data_seed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # the exact right-hand side on the evaluation system never reads the training trajectories: one stands in for them
    eval_only = all(m.startswith("LieNLSD_RHS") and m.endswith("@eval") for m in args.methods.split(","))
    tr, x, t, u, xs, ts = protocol_data(args.pde, cfg, args.seed, dseed, 1 if eval_only else args.n_train, args.n)
    tr = torch.tensor(cr.corrupt(tr[:, 0].numpy(), args.noise, args.thin, args.seed, cr.TRAIN)[1],
                      dtype=torch.float32)[:, None]
    u, u_seen = cr.corrupt(u, args.noise, args.thin, dseed, cr.EVAL)
    t, ts = t[::args.thin], ts[:, ::args.thin]
    print(f"[{args.pde}] seed {args.seed}, data seed {dseed}: training {tuple(tr.shape)}, evaluation {u.shape}, "
          f"class dimension {K_CLASS[args.pde]}, degree {DEGREE[args.pde]}, noise {args.noise:g}, thin {args.thin}, "
          f"{dev}", flush=True)
    out = dict(pde=args.pde, seed=args.seed, data_seed=dseed, u=u, xs=xs, ts=ts, noise=args.noise, thin=args.thin)
    done = []
    extra = {}
    if os.path.exists(args.out):
        # resume / add methods: keep what the file already holds, same evaluation trajectories only
        old = dict(np.load(args.out, allow_pickle=True))
        if old["u"].shape != u.shape or not np.allclose(old["u"], u, atol=1e-6 * np.abs(u).max()):
            raise SystemExit(f"{args.out} holds other evaluation trajectories")
        if float(old.get("noise", 0.0)) != args.noise or int(old.get("thin", 1)) != args.thin:
            raise SystemExit(f"{args.out} holds another corruption (noise {old.get('noise', 0.0)}, thin "
                             f"{old.get('thin', 1)})")
        out.update(old)
        done = [str(m) for m in old["methods"]]
    for name in args.methods.split(","):
        if args.untrained and name.partition("@")[0] in ("LieNLSD", "LieNLSD_RHS", "LieNLSD_RHS_spec"):
            print(f"  {name}: no training, no untrained null", flush=True)
            continue
        if name in done:
            print(f"  {name}: already in {args.out}", flush=True)
            continue
        t0 = time.time()
        try:
            fields = run_method(name, args.pde, tr, x, t, u, xs, ts, cfg, dev, extra, u_seen)
        except Exception:
            print(f"  {name}: FAILED\n{traceback.format_exc()}", flush=True)
            continue
        out[f"fields_{name}"] = fields.astype(np.float32)
        out[f"time_{name}"] = time.time() - t0
        if name in ("LIG", "LIG_k"):
            out[f"epochs_{name}"] = cfg.epochs_lig
        if name in extra:
            out[f"nown_{name}"] = extra[name]["n_own"]
            out[f"diag_{name}"] = json.dumps(extra[name]["diag"])
            print(f"  {name}: own count {extra[name]['n_own']}; " + json.dumps(
                {k: v for k, v in extra[name]["diag"].items() if k != "fit"}), flush=True)
            if "fit" in extra[name]["diag"]:
                fit = extra[name]["diag"]["fit"]
                print(f"  {name}: F {fit['optimizer']}, {fit['epochs']} epochs, best epoch {fit['best_epoch']}, "
                      f"held-out relative MSE {fit['val_rel_mse']:.3e}", flush=True)
        done.append(name)
        out["methods"] = np.array(done)
        np.savez(args.out, **out)                 # after every method: a later crash loses nothing
        print(f"  {name}: {fields.shape[1]} generators, {time.time() - t0:.0f} s; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
