"""Fresh-seed confirmation (F1, F2): FINS through the protocol's trained operators on evaluation data seeds 2000 + s.
Each operator is checked first: the same command on the protocol data (1000 + s) must reproduce the eigenvalues of
the protocol file fins_{pde}_s{s}.npz; a mismatch (wrong operator file or flags) stops before any fresh run.
Operators are loaded, never trained (no --train-if-missing).

  python run_fresh.py [--pdes burgers,heat,kdv] [--seeds 42,43,44] [--out fresh]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = {"burgers": ["exp40_window.py", "--pde", "burgers", "--op", "cno", "--n", "10"],
       "heat": ["exp40_window.py", "--pde", "heat", "--op", "cno", "--smooth", "--only-L", "1", "--n", "10",
                "--tag", "_smooth_L1"],
       "kdv": ["exp39_localised.py", "--pde", "kdv", "--op", "stepper", "--ic", "periodic", "--train-ic", "mixed",
               "--n", "10"]}


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen([sys.executable, "-u", *cmd], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print("  " + line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdes", default="burgers,heat,kdv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ref-dir", default=os.environ.get("FINS_RESULTS_DIR", "."), help="protocol files")
    ap.add_argument("--out", default="fresh", help="folder under FINS_RESULTS_DIR")
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()
    out = os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), args.out)
    os.makedirs(out, exist_ok=True)
    for pde in args.pdes.split(","):
        for s in (int(x) for x in args.seeds.split(",")):
            ref = os.path.join(args.ref_dir, f"fins_{pde}_s{s}.npz")
            if not os.path.exists(ref):
                raise SystemExit(f"protocol file {ref} not found (--ref-dir)")
            chk = os.path.join(out, f"check_{pde}_s{s}.npz")
            run(CMD[pde] + ["--seed", str(s), "--data-seed", str(1000 + s), "--save", chk])
            a, b = np.load(ref)["lam"], np.load(chk)["lam"]
            n = min(len(a), len(b))
            rel = float(np.max(np.abs(a[:n] - b[:n]) / np.abs(a[:n])))
            print(f"[{pde} s{s}] protocol eigenvalues reproduced to {rel:.1e} (tolerance {args.rtol:.0e})", flush=True)
            if rel > args.rtol:
                raise SystemExit(f"{pde} s{s}: the operator or the flags differ from the protocol run")
            run(CMD[pde] + ["--seed", str(s), "--data-seed", str(2000 + s), "--save",
                            os.path.join(out, f"fins_{pde}_s{s}.npz")])


if __name__ == "__main__":
    main()
