"""Analysis of a trained exp45 network (m > 6): what the directions of its span are, and the span's spectrum against
the evaluation horizon (exact operator).

  --extra: for each eigen-direction of the span (read out on training pairs, saved as W), the share of its
           characteristic inside the algebra and inside the algebra plus phi = polynomial in (x, y) of degree <= 2,
           and the energy shares of its four field components (held-out probe points);
  --horizons K1,K2,...: span eigenvalues on training and held-out pairs and the six lowest directions' missing
           dimensions at each horizon.

  python exp45_span_analysis.py --stem results/neural2d/exact_m8_s42 --data-seed 4042 --eval-seed 5042 --extra
  python exp45_span_analysis.py --stem dev_m8_s0 --data-seed 100 --eval-seed 101 --horizons 8,16,24,32
"""

import argparse

import numpy as np
import torch

import exp45_neural_2d as e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True, help="path without extension of the run's .pt")
    ap.add_argument("--data-seed", type=int, required=True)
    ap.add_argument("--eval-seed", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsub", type=int, default=3)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--horizons", default=None)
    args = ap.parse_args()
    ck = torch.load(args.stem + ".pt", weights_only=False)
    g = e.Grid()
    dtr = e.Data(g, e.trajectories(g, args.n_train, args.data_seed, args.nt, args.dt, args.nsub), args.dt)
    dev = e.Data(g, e.trajectories(g, args.n_eval, args.eval_seed, args.nt, args.dt, args.nsub), args.dt, dtr.s)
    net = e.GenNet(ck["m"], ck["width"], ck["head"])
    net.load_state_dict(ck["state"])
    net = net.double()
    probe = e.Probe(dev)
    F, Qc = probe.net(net)
    m = ck["m"]
    if args.extra:
        W = ck["W"]
        Fm, Qm = W.T @ F.reshape(m, -1), W.T @ Qc
        nm = len(e.MONS)
        phi_xy = [i for i, (a, b, c, d) in enumerate(e.MONS) if c == 0 and d == 0]
        C = np.zeros((4 * nm, len(phi_xy)))
        for k, i in enumerate(phi_xy):
            C[3 * nm + i, k] = 1.0
        _, Qphi = probe.poly(C)
        alg, both = e.orth(probe.true_q), e.orth(np.vstack([probe.true_q, Qphi]))
        for j in range(m):
            q, f = Qm[j], Fm[j].reshape(4, -1)
            share = (f ** 2).sum(1) / (f ** 2).sum()
            print(f"direction {j + 1}: in the algebra {np.linalg.norm(alg.T @ q) / np.linalg.norm(q):.3f}, in the "
                  f"algebra + phi(x, y) quadratics {np.linalg.norm(both.T @ q) / np.linalg.norm(q):.3f}; field energy "
                  "xi/eta/tau/phi " + " ".join(f"{v:.2f}" for v in share))
    if args.horizons:
        op = e.ExactOp(g, args.dt, args.nsub)
        for K in (int(k) for k in args.horizons.split(",")):
            pt, pe = e.pairs_of(args.n_train, args.nt, K, 8), e.pairs_of(args.n_eval, args.nt, K, 8)
            N, DA, G0, trBB = e.pencil(op, dtr, e.chars_poly, pt[::2], K, 2)
            cmu = trBB / np.trace(G0)
            lam = e.gsolve(N, DA + cmu * G0)[0]
            Nn, DAn, G0n, _ = e.pencil(op, dtr, e.chars_net(net), pt, K, 4)
            evf, W = e.gsolve(Nn, DAn + cmu * G0n)
            Ne, DAe, G0e, _ = e.pencil(op, dev, e.chars_net(net), pe, K, 4)
            eve = e.gsolve(Ne, DAe + cmu * G0e)[0]
            sc6 = probe.scores(W[:, :6].T @ F.reshape(m, -1), W[:, :6].T @ Qc)
            print(f"K {K}: relative change {np.sqrt(cmu):.2f}; polynomial lowest " + " ".join(f"{v:.1e}" for v in lam[:9]))
            print("   span, training: " + " ".join(f"{v:.1e}" for v in evf) + f"  (lambda7 / lambda6 {evf[6] / evf[5]:.1f})")
            print("   span, held-out: " + " ".join(f"{v:.1e}" for v in eve) + f"  (lambda7 / lambda6 {eve[6] / eve[5]:.1f})")
            print(f"   six lowest (training readout): missing (characteristics) {sc6['missing_q']:.3f}, (fields) "
                  f"{sc6['missing_f']:.3f}", flush=True)


if __name__ == "__main__":
    main()
