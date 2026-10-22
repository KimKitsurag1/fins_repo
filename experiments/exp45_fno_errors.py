"""Rollout errors of saved exp45 FNO2d steppers on several data sets, overall and per trajectory (float64).

  python exp45_fno_errors.py --seeds 42,43,44 --sets 4000:16,5000:8,8000:8
Set a:n is data seed a + s with n trajectories (4000 + s FINS training, 5000 + s held-out, others unseen).
"""

import argparse
import os

import torch

import exp45_neural_2d as e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--sets", default="4000:16,5000:8,8000:8")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="checkpoint tag (exp45 --fno-tag)")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = e.Grid()
    for s in (int(x) for x in args.seeds.split(",")):
        path = os.path.join(e.RESULTS, "neural2d", f"fno2d_pme_s{s}{args.tag}.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = e.FNO2d(**ck["cfg"])
        model.load_state_dict(ck["state"])
        model = model.double().to(dev).eval()
        for item in args.sets.split(","):
            base, n = (int(v) for v in item.split(":"))
            d = e.Data(g, e.trajectories(g, n, base + s, 51, 0.02, 3), 0.02).cast(torch.float64, dev)
            e1, eK, per = e.operator_errors(model, d, args.K)
            print(f"FNO s{s}, data seed {base + s} ({n}): 1 step {e1:.6f}, per trajectory " + " ".join(f"{v:.4f}" for v in per)
                  + f"; K steps {eK:.6f}; u max {float(d.u.max()):.3f}", flush=True)


if __name__ == "__main__":
    main()
