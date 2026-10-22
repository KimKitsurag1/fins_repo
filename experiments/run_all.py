"""One worker over the CPU jobs of exp52 (Benjamin-Ono), exp53 (Kuramoto-Sivashinsky), exp51 (noise study) and the
untrained baselines of exp54, in that order; GPU jobs (weak rows of KdV and KS, learned operators, trained exp54) are
not included. Start one worker per core; each imports torch once (~0.6 GB). Claims make concurrent workers safe.

Run:  python run_all.py [--threads 1]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402
import exp52_bo as e52                                     # noqa: E402
import exp53_ks as e53                                     # noqa: E402
import exp54_baselines_new as e54                          # noqa: E402

SEEDS = list(range(42, 52))
PARTS_KS = "pointwise,two-stage,two-stage-h05,lienlsd,lienlsd-f"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    root = os.environ.get("FINS_RESULTS_DIR", "results")
    todo = (e52.jobs(list(e52.CONDS), SEEDS, "weak,pointwise,two-stage,two-stage-h05,lienlsd-f".split(","),
                     os.path.join(root, "bo_runs"))
            + e53.jobs(list(e53.CONDS), SEEDS, PARTS_KS.split(","),
                       os.path.join(root, "ks_runs"))
            + e51.jobs(list(e51.CONDS), ["heat", "kdv"], SEEDS,
                       "weak,pointwise,two-stage,two-stage-h05,lienlsd,lienlsd-f".split(","),
                       os.path.join(root, "noise_weak_runs"))
            + e54.jobs(["bo", "ks"], [42, 43, 44], ["null"], root))
    e51.run(args, todo)


if __name__ == "__main__":
    main()
