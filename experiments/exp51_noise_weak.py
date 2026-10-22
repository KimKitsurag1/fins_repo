"""Experiment 51 -- noise study with the weak-form criterion (README_EXPERIMENTS.md, pre-registration of 06.10).

Heat (smooth data, L = 1, four-period windows, exact heat kernel) and KdV (periodic solitons on the torus, exact
solver); Gaussian noise of sigma times each trajectory's standard deviation (corrupt.py), sigma in CONDS; seeds 42-51,
evaluation data 1000 + s. Every method sees the same corrupted snapshots; generators are scored on the clean ones
(exp42's metric: band [0.2, 0.8] L, times 8 .. Nt - 8, m = k):
  weak        FINS in weak form (exp49): class tau(t), xi(x, t), phi(x, t, u); K = 16, a start at every snapshot,
              two temporal bumps (p = 2), spatial modes from the data's bandwidth (--wx auto, factor WX_FACTOR);
              also scored with its own count (largest eigenvalue ratio, no ground truth);
  pointwise   FINS pointwise, exp48's exact-operator rows (exp40 heat-kernel window / exp39 exact solver, u_t by
              fourth-order differences of the corrupted snapshots, K = 16 every 16 snapshots from snapshot 2);
  two-stage   weak SINDy then Lie's algorithm (exp50): library u^a d_x^m u (a <= 2, m <= 3) and a constant, 200
              subdomains of half-width H of each extent per trajectory, threshold 0.1 of the largest column-normalised
              coefficient; hard readout (the null space: the identified equation's algebra, its own count) and soft
              (the k smallest singular vectors); H = 0.2 (chosen on the development seed) and H = 0.05 (PySINDy's
              default subdomain, a sensitivity row);
  LieNLSD     exp41: the exact right-hand side with the determining system on the evaluation data
              (LieNLSD_RHS@eval, every seed) and the network F fitted to the corrupted training data as published
              (LieNLSD_F, seeds 42-44 at sigma >= 3e-2: about 8 minutes per fit; clean, 1e-3 and 1e-2 are exp48's);
  weak-learned  FINS in weak form through the KdV FNO steppers of exp39 / exp48 (seeds 42-44, trained on the
              equally corrupted training data: clean, 1e-3, 1e-2), FINS_RESULTS_DIR.

Run:  python exp51_noise_weak.py run [--conds clean,n1e-2] [--pdes heat,kdv] [--seeds 42,...,51] [--parts weak,...]
      python exp51_noise_weak.py summary
Workers may run side by side: each claims a job by creating <output>.claim with the boot id and its pid; a claim
from before a reboot or from a dead process is taken over. Outputs: --runs (npz, logs), --out (scores, summary).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp42_tables as e42                                 # noqa: E402
import exp49_weak as e49                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = {"clean": 0.0, "n1e-3": 1e-3, "n1e-2": 1e-2, "n3e-2": 3e-2, "n5e-2": 5e-2, "n1e-1": 1e-1}
SEEDS = ",".join(str(s) for s in range(42, 52))
PARTS = "weak,pointwise,two-stage,two-stage-h05,lienlsd,lienlsd-f,weak-learned"
FIT_SEEDS = (42, 43, 44)
TRIVIAL = ("∂_x", "∂_t")                      # given by any shift-equivariant, autonomous operator
FIT_CONDS = ("n3e-2", "n5e-2", "n1e-1")         # LieNLSD_F at clean, 1e-3, 1e-2: exp48's runs (same data and draws)
LEARNED = {"clean": "", "n1e-3": "_n1e-3", "n1e-2": "_n1e-2"}   # KdV steppers of exp39 / exp48 (tag per condition)
WX_FACTOR = 2.0
U_ATOL = 1e-4                                   # files of one seed hold the same clean data to this fraction of max |u|
NX = {"heat": 64, "kdv": 128}
NT0 = 128


def jobs(conds, pdes, seeds, parts, runs):
    out = []
    for c in conds:
        sigma = CONDS[c]
        d = os.path.join(runs, c)
        nz = ["--noise", f"{sigma:g}"]
        for seed in seeds:
            ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
            for pde in pdes:
                if "weak" in parts:
                    out.append((os.path.join(d, f"weak_{pde}_s{seed}.npz"),
                                ["exp49_weak.py", "--pde", pde, "--op", "exact", *ds, *nz, "--K", "16", "--stride",
                                 "1", "--wt", "2", "--wp", "2", "--wx", "auto", "--wx-factor", f"{WX_FACTOR:g}",
                                 "--save", os.path.join(d, f"weak_{pde}_s{seed}.npz")]))
                if "pointwise" in parts:
                    f = os.path.join(d, f"pointwise_{pde}_s{seed}.npz")
                    win = ["--K", "16", "--every", "16", "--thin", "1", "--interior"]
                    if pde == "heat":
                        cmd = ["exp40_window.py", "--pde", "heat", "--op", "heat-kernel", "--smooth", "--only-L", "1",
                               "--n", "10", *ds, *win, *nz, "--save", f]
                    else:
                        cmd = ["exp39_localised.py", "--pde", "kdv", "--op", "exact", "--ic", "periodic", "--n", "10",
                               *ds, *win, "--ut", "fd", *nz, "--save", f]
                    out.append((f, cmd))
                for part, H in (("two-stage", "0.2"), ("two-stage-h05", "0.05")):
                    if part in parts:
                        pre = os.path.join(d, f"{part}_{pde}_s{seed}")
                        out.append((pre + "_soft.npz", ["exp50_two_stage.py", "--pde", pde, *ds, *nz, "--H", H,
                                                        "--K", "200", "--rel", "0.1", "--out", pre]))
                if "weak-learned" in parts and pde == "kdv" and seed in FIT_SEEDS and c in LEARNED:
                    f = os.path.join(d, f"weak-learned_{pde}_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", "--pde", pde, "--op", "learned", *ds, "--tag", LEARNED[c], *nz,
                                    "--K", "16", "--stride", "1", "--wt", "2", "--wp", "2", "--wx", "auto",
                                    "--wx-factor", f"{WX_FACTOR:g}", "--save", f]))
                for part, methods in (("lienlsd", "LieNLSD_RHS@eval"), ("lienlsd-f", "LieNLSD_F")):
                    if part in parts and (part == "lienlsd" or (seed in FIT_SEEDS and c in FIT_CONDS)):
                        f = os.path.join(d, f"{part}_{pde}_s{seed}.npz")
                        out.append((f, ["exp41_baselines.py", "--pde", pde, "--seed", str(seed), "--methods", methods,
                                        "--lienlsd-fit-points", str(10 * NX[pde] * NT0),
                                        "--lienlsd-val-points", str(2 * NX[pde] * NT0), *nz, "--out", f]))
    return out


def done(f):
    return os.path.exists(f) and not os.path.exists(f + ".claim")


def boot_id():
    try:
        return open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return "unknown"


def stale(claim):
    """A claim written before the last reboot, or by a process that no longer runs, holds nothing."""
    try:
        bid, pid = open(claim).read().split()
    except (OSError, ValueError):
        return False
    if bid != boot_id():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


def run(args, todo=None):
    if todo is None:
        todo = jobs(args.conds.split(","), args.pdes.split(","), [int(x) for x in args.seeds.split(",")],
                    args.parts.split(","), args.runs)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads))
    failed = set()
    while True:
        left = [j for j in todo if j[0] not in failed and not done(j[0])]
        if not left:
            break
        ran = False
        for f, cmd in left:
            os.makedirs(os.path.join(os.path.dirname(f), "logs"), exist_ok=True)
            claim = f + ".claim"
            if done(f):
                continue
            if os.path.exists(claim) and stale(claim):
                os.remove(claim)
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{boot_id()} {os.getpid()}".encode())
            os.close(fd)
            log = os.path.join(os.path.dirname(f), "logs", os.path.basename(f).replace("_soft", "")
                               .replace("_oracle.json", ".log").replace(".npz", ".log").replace(".json", ".log"))
            t0 = time.time()
            try:
                with open(log, "a") as fh:
                    fh.write("$ " + " ".join(cmd) + "\n")
                    fh.flush()
                    rc = subprocess.call([sys.executable, "-u", *cmd], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                         env=env)
            finally:
                os.remove(claim)
            ok = rc == 0 and done(f)
            if not ok:
                failed.add(f)
            ran = True
            print(f"{'ok' if ok else 'FAILED'} {time.time() - t0:6.0f} s  {f}", flush=True)
            break
        if not ran:
            time.sleep(20)


# ---------------------------------------------------------------------------------------------------------------------
# scoring

def fins_like(z, m=None):
    """Fields [m, 3, N, Nx, Nt] of the leading m generators of an exp49 / exp50 / exp39-40 file (default its k)."""
    if m is None:
        return e42.fins_fields(z)
    zz = {key: z[key] for key in z.files}
    zz["k"] = m
    return e42.fins_fields(zz)


def rows_of(d, pde, seed):
    """{row label: fields} for one seed, plus the clean data the files agree on."""
    out, files = {}, []
    for tag, label in (("weak", "FINS weak"), ("pointwise", "FINS pointwise"),
                       ("weak-learned", "FINS weak, learned operator"),
                       ("pointwise-learned", "FINS pointwise, learned operator"),
                       ("weak-learned-attractor", "FINS weak, learned on zero-mean data"),
                       ("pointwise-learned-attractor", "FINS pointwise, learned on zero-mean data")):
        p = os.path.join(d, f"{tag}_{pde}_s{seed}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out[label] = fins_like(z)
            if tag.startswith("weak"):
                out[label + ", own count"] = fins_like(z, e49.own_count(z["lam"]))
            files.append(z)
    for tag, label in (("two-stage", "two-stage, H 0.2"), ("two-stage-h05", "two-stage, H 0.05"),
                       ("two-stage-o4", "two-stage, H 0.2, library to u_xxxx"),
                       ("two-stage-h05-o4", "two-stage, H 0.05, library to u_xxxx"),
                       ("wsindy", "WSINDy, evaluation data"), ("wsindy-train", "WSINDy, training data"),
                       ("kreider-sheet", "Kreider et al., solutions"),
                       ("kreider-surface", "Kreider et al., equation surface")):
        for kind in ("soft", "hard"):
            p = os.path.join(d, f"{tag}_{pde}_s{seed}_{kind}.npz")
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                out[f"{label}, {kind}" + (" (own count)" if kind == "hard" else "")] = fins_like(z)
                files.append(z)
    for tag in ("lienlsd", "lienlsd-f", "baselines", "baselines-lig", "baselines-null", "baselines-lig-null"):
        p = os.path.join(d, f"{tag}_{pde}_s{seed}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            for m in z["methods"]:
                lab = f"{m}, untrained" if tag.endswith("-null") else str(m)
                out[lab] = np.moveaxis(z[f"fields_{m}"].astype(np.float64), 0, 2)
                if f"nown_{m}" in z.files:
                    out[f"{lab}, own count"] = out[lab][:int(z[f"nown_{m}"])]
            files.append(z)
    if not files:
        return out, None
    u = files[0]["u"]
    for z in files[1:]:
        # the same trajectories up to the solver's roundoff on another device (KdV's weak rows ran on a Colab GPU)
        if z["u"].shape != u.shape or not np.allclose(z["u"], u, atol=U_ATOL * np.abs(u).max()):
            raise SystemExit(f"{d} {pde} s{seed}: evaluation trajectories differ between files")
    return out, files[0]


def score_cond(d, pdes, seeds, cfg):
    res = {}
    for pde in pdes:
        for seed in seeds:
            fields, ref = rows_of(d, pde, seed)
            if ref is None:
                continue
            u, xs, ts = ref["u"], ref["xs"], ref["ts"]
            Nx = u.shape[1]
            frac = np.arange(Nx) / Nx
            keep = (frac >= 0.2) & (frac <= 0.8)
            L = (xs[0, 1] - xs[0, 0]) * Nx
            w = np.array([1 / L, 1 / ts[:, -1].max(), 1 / u.std()])
            names, gt = e42.ground_truth(pde, u, xs, ts, cfg)
            G = e42.vectors(gt, keep, w, 8)
            nf = e42.FINITE[pde]
            # the finite algebra's generators other than the translations, each reduced to its part orthogonal to
            # every other generator: what an autonomous, shift-equivariant operator does not give for free
            gps = e42.novel_parts(G[:nf])
            NT = np.stack([g for g, nm in zip(gps, names[:nf]) if nm not in TRIVIAL])
            for label, f in fields.items():
                if f.shape[2:] != u.shape:
                    raise SystemExit(f"{d} {pde} s{seed} {label}: evaluation data of another shape")
                D = e42.vectors(f, keep, w, 8)
                D = np.where(np.isfinite(D), D, 0.0)
                r = {"finite": e42.score(D, G[:nf], gps), "non-trivial": e42.score(D, NT)}
                if len(G) > nf:
                    r["full"] = e42.score(D, G, e42.novel_parts(G))
                for cv, v in r.items():
                    slot = res.setdefault(pde, {}).setdefault(label, {}).setdefault(cv, {})
                    for key in ("missing", "spurious", "k"):
                        slot.setdefault(key, {})[str(seed)] = float(v[key])
                    slot.setdefault("contain", {})[str(seed)] = np.asarray(v["contain"]).tolist()
                    slot.setdefault("novel", {})[str(seed)] = np.asarray(v["novel"]).tolist()
            res[pde]["_names"] = list(names)
    return res


def exp48_rows(c, pdes):
    """exp48's rows on the same data and draws (seeds 42-44): LieNLSD as published at clean, 1e-3, 1e-2, and FINS
    pointwise through the learned operators (results/noise_thin/<cond>/learned/scores_<pde>.json)."""
    out = {}
    base = os.path.join(HERE, "results", "noise_thin", c)
    p = os.path.join(base, "scores.json")
    if os.path.exists(p):
        js = json.load(open(p))
        for pde in pdes:
            for lab in ("LieNLSD_F", "LieNLSD_F, own count"):
                if lab in js.get(pde, {}):
                    out.setdefault(pde, {})[lab + " (exp48)"] = js[pde][lab]
    for pde in pdes:
        p = os.path.join(base, "learned", f"scores_{pde}.json")
        if os.path.exists(p):
            js = json.load(open(p))[pde]
            out.setdefault(pde, {})["FINS pointwise, learned operator (exp48)"] = js["FINS"]
    return out


def summary(args, conds=CONDS):
    cfg = e31.ExperimentConfig()
    pdes, seeds = args.pdes.split(","), [int(x) for x in args.seeds.split(",")]
    allres = {}
    for c in conds:
        d = os.path.join(args.runs, c)
        if not glob.glob(os.path.join(d, "*.npz")):
            continue
        allres[c] = score_cond(d, pdes, seeds, cfg)
        if args.exp48:
            for pde, rows in exp48_rows(c, pdes).items():
                for lab, r in rows.items():
                    # exp48's JSON: per-seed lists in seed order 42, 43, 44
                    allres[c].setdefault(pde, {})[lab] = {
                        cv: {"missing": dict(zip(map(str, FIT_SEEDS), v["missing"]))} for cv, v in r.items()}
        os.makedirs(os.path.join(args.out, c, "logs"), exist_ok=True)
        with open(os.path.join(args.out, c, "scores.json"), "w") as fh:
            json.dump(allres[c], fh, indent=1)
        for p in glob.glob(os.path.join(d, "logs", "*.log")):
            subprocess.check_call(["cp", p, os.path.join(args.out, c, "logs")])
    conds = list(allres)
    lines = []
    for pde in pdes:
        for cv in ("finite", "non-trivial", "full"):
            labels = []
            for c in conds:
                for lab, r in allres[c].get(pde, {}).items():
                    if not lab.startswith("_") and cv in r and lab not in labels:
                        labels.append(lab)
            if not labels:
                continue
            lines.append(f"## {pde}, {cv} algebra: missing dimensions (mean +/- sd over seeds, n seeds)\n")
            lines.append("| row | " + " | ".join(conds) + " |")
            lines.append("|---" * (len(conds) + 1) + "|")
            for lab in labels:
                cells = []
                for c in conds:
                    v = allres[c].get(pde, {}).get(lab, {}).get(cv, {}).get("missing", {})
                    vals = np.array(list(v.values()))
                    cells.append(f"{vals.mean():.3f} +/- {vals.std():.3f} ({len(vals)})" if len(vals) else "")
                lines.append(f"| {lab} | " + " | ".join(cells) + " |")
            lines.append("")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(allres, fh, indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--pdes", default="heat,kdv")
    ap.add_argument("--seeds", default=SEEDS)
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "noise_weak_runs"))
    ap.add_argument("--out", default=os.path.join("results", "noise_weak"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--exp48", action="store_true", help="summary: add exp48's rows (seeds 42-44, same data)")
    args = ap.parse_args()
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(HERE, p) for p in (args.runs, args.out))
    (run if args.what == "run" else summary)(args)


if __name__ == "__main__":
    main()
