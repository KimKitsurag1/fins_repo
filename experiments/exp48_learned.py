"""Experiment 48, learned arm -- FINS through learned operators under noise and thinning (README_EXPERIMENTS.md).

exp48_noise_thinning.py compared the criteria with perfect models and ran LieNLSD as published (network F fitted to
the corrupted training data). This arm gives FINS's row with learned operators on the same corrupted data, so that
the two methods are compared with the models each of them learns: FINS a solution operator, LieNLSD a network for
u_t. Per condition, seeds 42-44, evaluation data 1000 + s, exp48's windows (snapshots 2 .. Nt - 3, the same physical
horizon at every thinning) and u_t (five-point differences of the corrupted snapshots; for KdV in place of the
protocol's solver stencil):
  clean     the protocol's operators, loaded (S_theta_heat_s{s}_g64x128_cno_smooth_L1.pt,
            S_theta_kdv_s{s}_mixed_stepper.pt in FINS_RESULTS_DIR), no training;
  clean-rt  operators retrained on the clean training data with the protocol's settings: the control for the
            corrupted conditions when those are trained on another device than the protocol's operators;
  others    operators retrained on the equally corrupted training data (protocol settings, 400 epochs; the heat
            CNO is cropped by 6 x thin points per training step, unrolled over at most 16 original steps (4 kept
            steps, 2 at s = 8, where 4 would leave no target in the 256-point window) and scored on the protocol's
            region, 96 points from each window edge).
The commands are those of the Colab cells "exp48". Generators are scored on the clean evaluation trajectories (exp42,
--t-margin 8 // s); the perfect-model rows come from exp48's summary.json.

Run:  python exp48_learned.py run --pdes kdv --conds n1e-4,t4 [--seeds 42,43,44] [--threads 1]
      python exp48_learned.py summary
Workers may run side by side: each claims a job by creating <output>.claim with the boot id and its pid; a claim
from before a reboot or from a dead process is taken over (--reset-claims removes every claim, with no worker
running). Checkpoints go to FINS_RESULTS_DIR with the condition in their tag, and training writes a partial state
every 10 epochs, so a stopped run resumes where it was.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = {"clean": (0.0, 1), "clean-rt": (0.0, 1), "n1e-4": (1e-4, 1), "n3e-4": (3e-4, 1), "n1e-3": (1e-3, 1),
         "n1e-2": (1e-2, 1), "t2": (0.0, 2), "t4": (0.0, 4), "t8": (0.0, 8)}
PERFECT = ["FINS, exact operator", "LieNLSD_RHS@eval", "LieNLSD_RHS_spec@eval", "LieNLSD_F", "LieNLSD_F@eval",
           "null: Gaussian coefficients"]


def learned_jobs(conds, pdes, seeds, runs):
    out = []
    for c in conds:
        sigma, s = CONDS[c]
        d = os.path.join(runs, c)
        corr = ["--K", str(16 // s), "--every", str(16 // s), "--noise", f"{sigma:g}", "--thin", str(s)]
        for seed in seeds:
            ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
            for pde in pdes:
                f = os.path.join(d, f"fins_{pde}_s{seed}.npz")
                if pde == "heat":
                    base = ["exp40_window.py", "--pde", "heat", "--op", "cno", "--smooth", "--only-L", "1", "--n", "10"]
                    if c == "clean":
                        cmd = [*base, "--tag", "_smooth_L1", *ds, "--interior", "--save", f]
                    else:
                        # unroll x thin <= 16 original steps: at s = 8 an unroll of 4 would crop 4 x 48 points per
                        # side of the 256-point window, leaving no target (the protocol's 4 is kept for s <= 4)
                        cmd = [*base, "--tag", f"_smooth_L1_{c}", "--train-if-missing", *ds, "--score-margin", "96",
                               "--train-margin", str(6 * s), "--unroll", str(min(4, 16 // s)), *corr, "--interior",
                               "--save", f]
                else:
                    base = ["exp39_localised.py", "--pde", "kdv", "--op", "stepper", "--ic", "periodic", "--train-ic",
                            "mixed", "--n", "10"]
                    if c == "clean":
                        cmd = [*base, *ds, "--ut", "fd", "--interior", "--save", f]
                    else:
                        cmd = [*base, "--tag", f"_{c}", "--train-if-missing", *ds, "--ut", "fd", *corr, "--interior",
                               "--save", f]
                out.append((f, cmd))
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
        return False                              # being written, or an old empty claim: leave it
    if bid != boot_id():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


def run(args):
    todo = learned_jobs(args.conds.split(","), args.pdes.split(","), [int(x) for x in args.seeds.split(",")],
                        args.runs)
    if args.reset_claims:
        for p in glob.glob(os.path.join(args.runs, "*", "*.claim")):
            os.remove(p)
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
                continue                          # another worker holds this job
            os.write(fd, f"{boot_id()} {os.getpid()}".encode())
            os.close(fd)
            log = os.path.join(os.path.dirname(f), "logs", os.path.basename(f).replace(".npz", ".log"))
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
            break                                 # rescan: claims may have been released meanwhile
        if not ran:
            time.sleep(20)


def summary(args):
    perfect = json.load(open(os.path.join(args.out, "summary.json")))
    seeds = [int(x) for x in args.seeds.split(",")]
    rows, seen, dt = {}, {}, {}
    for c, (sigma, s) in CONDS.items():
        d = os.path.join(args.runs, c)
        dst = os.path.join(args.out, c, "learned")
        # one score file per equation: runs of one equation (e.g. heat from Colab) never overwrite the other's scores,
        # and an equation whose npz files are gone is read back from its stored scores
        for pde in sorted({os.path.basename(p).split("_")[1] for p in glob.glob(os.path.join(d, "fins_*.npz"))}):
            have = [x for x in seeds if os.path.exists(os.path.join(d, f"fins_{pde}_s{x}.npz"))]
            js = os.path.join(dst, f"scores_{pde}.json")
            os.makedirs(os.path.join(dst, "logs"), exist_ok=True)
            with open(os.path.join(dst, f"tables_{pde}.md"), "w") as fh:
                subprocess.check_call([sys.executable, "-u", "exp42_tables.py", "--dir", d, "--pdes", pde,
                                       "--seeds", ",".join(map(str, have)), "--t-margin", str(max(1, 8 // s)),
                                       "--json", js], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            res = json.load(open(js))
            res[pde]["_seeds"] = have  # exp42 scores the seeds present in seed order; keep their labels
            with open(js, "w") as fh:
                json.dump(res, fh, indent=1)
            for p in glob.glob(os.path.join(d, "logs", f"fins_{pde}_*.log")):
                shutil.copy(p, os.path.join(dst, "logs"))
        for js in sorted(glob.glob(os.path.join(dst, "scores_*.json"))):
            for pde, tab in json.load(open(js)).items():
                if "FINS" not in tab:
                    continue
                seen[(pde, c)] = tab["_seeds"]
                for cv, v in tab["FINS"].items():
                    rows.setdefault(pde, {}).setdefault(f"FINS, learned operator [{cv}]", {})[c] = v["missing"]
                names = tab["_names"]
                dt[(pde, c)] = [cs[names.index("∂_t")] for cs in tab["FINS"]["finite"]["contain"]]
    for pde in rows:
        for lab, r in perfect.get(pde, {}).items():
            if any(lab.startswith(p + " [") or (p.startswith("null") and lab.startswith(p)) for p in PERFECT):
                rows[pde][lab] = {c: v for c, v in r.items() if c != "protocol"}
    with open(os.path.join(args.out, "summary_learned.json"), "w") as fh:
        json.dump({"rows": rows, "seeds": {f"{p} {c}": v for (p, c), v in seen.items()},
                   "dt_contain": {f"{p} {c}": v for (p, c), v in dt.items()}}, fh, indent=1)
    conds = [c for c in CONDS if any(c in r for t in rows.values() for r in t.values())]
    with open(os.path.join(args.out, "summary_learned.md"), "w") as fh:
        for pde, tab in rows.items():
            fh.write(f"## {pde}: missing dimensions, m = k, mean +/- sd over seeds (n); per seed in the JSON; clean-rt: "
                     f"operators retrained on clean data (the perfect-model rows are those of clean)\n\n")
            fh.write("| row | " + " | ".join(conds) + " |\n" + "|---" * (len(conds) + 1) + "|\n")
            for lab, r in tab.items():
                cells = [f"{np.mean(r[c]):.2f} +/- {np.std(r[c]):.2f} ({len(r[c])})" if c in r else "" for c in conds]
                fh.write(f"| {lab} | " + " | ".join(cells) + " |\n")
            fh.write("\n")
        fh.write("## Registered predictions (README, learned arm), evaluated on the seeds present\n\n")
        for line in verdicts(rows, seen, dt, seeds):
            fh.write(line + "\n")
    print(open(os.path.join(args.out, "summary_learned.md")).read())


def verdicts(rows, seen, dt, seeds):
    """The registered predictions, literally; 'incomplete' while a condition lacks seeds."""
    def get(pde, lab, c):
        r = rows.get(pde, {}).get(lab, {})
        return r.get(c)

    def fins(pde, c):
        v = get(pde, "FINS, learned operator [finite]", c)
        return (np.array(v), seen[(pde, c)]) if v is not None else (None, [])

    def perfect(pde, lab, c, on):
        v = get(pde, lab, c)
        return None if v is None else np.array([v[seeds.index(x)] for x in on])

    def full(pde, c):
        return len(fins(pde, c)[1]) == len(seeds)

    fmt = lambda a: " / ".join(f"{x:.2f}" for x in a)
    out = []
    v, on = fins("heat", "clean")
    if v is not None:
        out.append(f"(C1) heat clean, protocol CNOs: {v.mean():.2f} (per seed {fmt(v)}) against 0.14 +/- 0.10 -> "
                   f"{'met' if abs(v.mean() - 0.14) <= 0.10 else 'NOT MET'}" + ("" if full("heat", "clean") else
                                                                                    " (incomplete)"))
    v, on = fins("kdv", "clean")
    if v is not None:
        out.append(f"(C1) KdV clean, protocol steppers: {v.mean():.2f} (per seed {fmt(v)}) against <= 0.05 -> "
                   f"{'met' if v.mean() <= 0.05 else 'NOT MET'}" + ("" if full("kdv", "clean") else " (incomplete)"))
    for pde, bound in (("kdv", None), ("heat", 0.15)):
        v, on = fins(pde, "clean-rt")
        if v is None:
            continue
        ok = (v <= 0.05).all() if pde == "kdv" else abs(v.mean() - 0.14) <= bound
        out.append(f"(LC) {pde} clean-rt: {v.mean():.2f} (per seed {fmt(v)}) -> {'met' if ok else 'NOT MET'}"
                   + ("" if full(pde, "clean-rt") else " (incomplete)"))
    for pde, tag in (("kdv", "LT1"), ("heat", "LT2")):
        parts, oks, comp = [], [], True
        ref, _ = fins(pde, "clean-rt")
        for c in ("t2", "t4", "t8"):
            v, on = fins(pde, c)
            if v is None:
                comp = False
                continue
            comp &= full(pde, c)
            F = perfect(pde, "LieNLSD_F [finite]", c, on)
            if pde == "kdv":
                oks += [v.mean() <= 0.10, bool((v < F).all())]
            else:
                oks += [ref is not None and abs(v.mean() - ref.mean()) <= 0.15, v.mean() < F.mean()]
            parts.append(f"{c}: {v.mean():.2f} ({fmt(v)}) vs F {F.mean():.2f} ({fmt(F)})")
        if parts:
            out.append(f"({tag}) {pde} thinning: " + "; ".join(parts) + (f"; clean-rt {ref.mean():.2f}"
                       if ref is not None and pde == "heat" else "") + f" -> {'met' if all(oks) else 'NOT MET'}"
                       + ("" if comp else " (incomplete)"))
    for pde, tag in (("heat", "LN1"), ("kdv", "LN2")):
        parts, oks, comp = [], [], True
        for c in ("n1e-4", "n3e-4", "n1e-3"):
            v, on = fins(pde, c)
            if v is None:
                comp = False
                continue
            comp &= full(pde, c)
            E = perfect(pde, "FINS, exact operator [finite]", c, on)
            oks.append(v.mean() >= E.mean() - 0.05 if pde == "heat" else abs(v.mean() - E.mean()) <= 0.15)
            parts.append(f"{c}: {v.mean():.2f} ({fmt(v)}) vs exact {E.mean():.2f}")
        if parts:
            out.append(f"({tag}) {pde} noise: " + "; ".join(parts) + f" -> {'met' if all(oks) else 'NOT MET'}"
                       + ("" if comp else " (incomplete)"))
    parts, oks, comp = [], [], True
    for pde, c, rule in (("kdv", "n1e-4", "fewer, per seed"), ("kdv", "n3e-4", "fewer, per seed"),
                         ("kdv", "n1e-2", "more, mean"), ("heat", "n1e-2", "more, mean")):
        v, on = fins(pde, c)
        if v is None:
            comp = False
            continue
        comp &= full(pde, c)
        F = perfect(pde, "LieNLSD_F [finite]", c, on)
        oks.append(bool((v < F).all()) if rule.startswith("fewer") else v.mean() > F.mean())
        parts.append(f"{pde} {c} ({rule}): {fmt(v)} vs F {fmt(F)}")
    if parts:
        out.append("(LN3) " + "; ".join(parts) + f" -> {'met' if all(oks) else 'NOT MET'}"
                   + ("" if comp else " (incomplete)"))
    parts, oks, comp = [], [], True
    for pde in ("heat", "kdv"):
        for c in ("n1e-3", "n1e-2"):
            if (pde, c) not in dt:
                comp = False
                continue
            comp &= full(pde, c)
            d_ = np.array(dt[(pde, c)])
            oks.append((d_ < 0.9).sum() >= 2)
            parts.append(f"{pde} {c}: d_t containment {fmt(d_)}")
    if parts:
        out.append("(LN4) " + "; ".join(parts) + f" -> {'met' if all(oks) else 'NOT MET'}"
                   + ("" if comp else " (incomplete)"))
    return out


def latex(args):
    """Rows of the paper's noise tables from summary_learned.json: means (Table 5) and means +/- sd (Table 11)."""
    rows = json.load(open(os.path.join(args.out, "summary_learned.json")))["rows"]
    order = ["clean", "n1e-4", "n3e-4", "n1e-3", "n1e-2", "t2", "t4", "t8"]
    for pde, cv, title in (("heat", "finite", "heat (6)"), ("heat", "full", "heat, class (10)"),
                           ("kdv", "finite", "KdV (4)")):
        r = rows.get(pde, {}).get(f"FINS, learned operator [{cv}]", {})
        mean = [f"{np.mean(r[c]):.2f}" if c in r else "--" for c in order]
        sd = [f"{np.mean(r[c]):.2f}\\,$\\pm$\\,{np.std(r[c]):.2f}" if c in r else "--" for c in order]
        print(f"% {title}; seeds per condition: " + ", ".join(f"{c} {len(r[c])}" for c in order if c in r))
        print("FINS, learned operator & " + " & ".join(mean) + " \\\\")
        print("FINS, learned operator & " + " & ".join(sd) + " \\\\")
        if "clean-rt" in r:
            print(f"% clean-rt (retrained on clean data): {np.mean(r['clean-rt']):.2f} +/- {np.std(r['clean-rt']):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary", "latex"])
    ap.add_argument("--conds", default="clean,n1e-4,t4")
    ap.add_argument("--pdes", default="heat,kdv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"),
                                                   "noise_thin_learned_runs"), help="npz files and logs")
    ap.add_argument("--out", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "noise_thin"),
                    help="exp48's summary folder: learned tables go to <cond>/learned/, the merged table to "
                         "summary_learned.md")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reset-claims", action="store_true")
    args = ap.parse_args()
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(HERE, p) for p in (args.runs, args.out))
    {"run": run, "summary": summary, "latex": latex}[args.what](args)


if __name__ == "__main__":
    main()
