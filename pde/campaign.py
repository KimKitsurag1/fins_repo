"""campaign — run the Table 1 campaign with an on-disk status check.

run_single_pde.py always recomputes everything; it has no notion of what
already exists. This driver treats the results directory as the source of truth:

  * a (pde, seed) pair is DONE when both its generator artifacts
    (generators_{pde}_seed{N}.npz) and its metrics json
    (results_{pde}_s{N}_{backend}.json) exist and load;
  * a pair is STALE when its artifacts exist but predate a change that
    invalidates them -- currently: the Burgers projective ground-truth fix
    (any Burgers artifact without a marker is stale), and the multi-sample
    block required by the closure metric of closure_metrics.py (artifacts without
    `u_ref` cannot be scored with --ace; they are reported, not re-run,
    since d_G / cos / rank are still valid for them);
  * everything else is PENDING and is run, one pair at a time, each pair
    writing its own artifacts immediately, so a killed session loses at
    most the pair in flight.

Run:  python campaign.py --status
      python campaign.py --run --pdes heat,burgers,kdv --seeds 42,43,44
      python campaign.py --run --only-stale
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def results_dir():
    return os.environ.get("FINS_RESULTS_DIR", HERE)


def status_of(pde, seed, backend="cno", want_n_sym=None, tag=""):
    npz = os.path.join(results_dir(), f"generators_{pde}{tag}_seed{seed}.npz")
    js = glob.glob(os.path.join(HERE, f"results_{pde}{tag}_s{seed}_*.json")) + \
        glob.glob(os.path.join(results_dir(),
                               f"results_{pde}{tag}_s{seed}_*.json"))
    have_npz, have_js = os.path.exists(npz), bool(js)
    if not (have_npz and have_js):
        return "PENDING", (f"npz={'ok' if have_npz else 'missing'} "
                           f"json={'ok' if have_js else 'missing'}")
    try:
        z = np.load(npz, allow_pickle=True)
        keys = set(z.files)
    except Exception as exc:
        return "PENDING", f"npz unreadable: {exc}"
    notes = []
    if pde == "burgers" and "gt_projective_corrected" not in keys:
        # artifacts written before the ground-truth fix carry the wrong v5
        return "STALE", "burgers GT predates the projective fix"
    if "u_ref" not in keys:
        notes.append("single-sample: --ace unavailable")
    # Generator COUNT per method. d_G carries a rank penalty
    # |k_A - k_B| * (pi/2)^2 driven by how many generators a method was
    # configured to emit, not by what it found, so runs with different
    # counts are not comparable. --quick silently sets cfg.n_sym = 8
    # against the default 6, which alone shifts d_G by up to 2.22.
    rec = int(z["n_sym"]) if "n_sym" in keys else None
    if want_n_sym is not None and rec is not None and rec != want_n_sym:
        return "STALE", (f"n_sym={rec} but the campaign asks for "
                         f"{want_n_sym}; d_G would not be comparable")
    if rec is None:
        notes.append("n_sym unrecorded")
    counts = {k[:-3]: int(z[k].shape[0]) for k in keys
              if k.endswith("_xi") and not k.startswith("gt")}
    notes.append("k=" + ",".join(f"{m[:4]}:{n}" for m, n in
                                 sorted(counts.items())))
    return "DONE", "; ".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only-stale", action="store_true")
    ap.add_argument("--pdes", default="heat,burgers,kdv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-sym", type=int, default=6,
                    help="generators per method; fixed across the campaign "
                         "so d_G stays comparable")
    ap.add_argument("--fins-lambda-b", type=float, default=0.0,
                    help="run the FINS-v2 constrained variant with this "
                         "b-share penalty (exp6: 30); artifacts get --fins-tag")
    ap.add_argument("--fins-anchor", default="deflated",
                    choices=["deflated", "raw"])
    ap.add_argument("--provided-ut", action="store_true",
                    help="KdV: solver RHS as u_t (diagnostic)")
    ap.add_argument("--surrogate-tangent", type=int, default=0,
                    help="tangent (Jacobian) targets per IC for the "
                         "surrogate; 0 = off")
    ap.add_argument("--fins-channel-norm", default="none",
                    choices=["none", "q"],
                    help="'q' normalises the residual by ||Q||, removing "
                         "the eta-channel collapse on Burgers/KdV")
    ap.add_argument("--fins-b-penalty", default="full",
                    choices=["full", "nonconst"],
                    help="use 'nonconst' when lambda_b is applied to "
                         "Burgers/KdV, whose Galilean boost is b = const")
    ap.add_argument("--fins-lambda-b-heat-only", action="store_true",
                    help="apply --fins-lambda-b to heat only: on Burgers "
                         "and KdV the Galilean boost is phi = const, i.e. a "
                         "PURE b term, and the superposition subalgebra the "
                         "penalty targets exists only for linear PDEs")
    ap.add_argument("--fins-tag", default="",
                    help="artifact suffix for a FINS variant, e.g. _v2")
    ap.add_argument("--extra", default="",
                    help="extra args forwarded to run_single_pde, "
                         "e.g. '--quick'")
    args = ap.parse_args()

    pdes = args.pdes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"results dir: {results_dir()}")
    print("k = generators emitted per method; runs whose counts differ are "
          "NOT comparable on d_G")
    todo = []
    for pde in pdes:
        for seed in seeds:
            st, note = status_of(pde, seed, want_n_sym=args.n_sym,
                                 tag=args.fins_tag)
            print(f"  {pde:<8}{args.fins_tag} seed {seed}: {st:<8} {note}")
            if args.only_stale:
                if st == "STALE":
                    todo.append((pde, seed))
            elif st in ("PENDING", "STALE"):
                todo.append((pde, seed))
    if not args.run:
        print(f"\n{len(todo)} pair(s) would run; add --run to start")
        return
    for pde, seed in todo:
        t0 = time.time()
        print(f"\n=== running {pde} seed {seed} ===", flush=True)
        cmd = [sys.executable, "-u", os.path.join(HERE, "run_single_pde.py"),
               "--pde", pde, "--seed", str(seed),
               "--n-sym", str(args.n_sym)] + (
            ["--fins-lambda-b",
             str(args.fins_lambda_b
                 if (pde == "heat" or not args.fins_lambda_b_heat_only)
                 else 0.0),
             "--fins-anchor", args.fins_anchor,
             "--fins-b-penalty", args.fins_b_penalty,
             "--fins-channel-norm", args.fins_channel_norm,
             "--surrogate-tangent", str(args.surrogate_tangent),
             *(["--provided-ut"] if (args.provided_ut and pde == "kdv") else []),
             "--fins-tag", args.fins_tag] if args.fins_tag else []
        ) + args.extra.split()
        p = subprocess.run(cmd, cwd=HERE)
        st, note = status_of(pde, seed, want_n_sym=args.n_sym,
                             tag=args.fins_tag)
        print(f"=== {pde}{args.fins_tag} seed {seed}: exit {p.returncode}, "
              f"now {st} "
              f"({note}), {time.time() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
