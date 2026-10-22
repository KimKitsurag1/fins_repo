"""Finite flows exp(s v) of generators through the exact 2D operator of exp45: does the equivariance defect grow with
the size of the transformation?

For a generator v with characteristic Q, the finite transformation of a snapshot w at time t is the solution at s of
dw/ds = P Q[w; t] (evolutionary form; u_t := the operator's own generator Lap(w^2 / 2); P the solver's circular 2/3
band, so the flow stays in the solver's state space). Away from the seam the coefficient functions are evaluated at
windowed coordinates x' = chi(x) x, y' = chi(y) y (chi = 1 for |x| <= inner, 0 beyond outer, C-infinity in between),
so every field is smooth and periodic; the window leaves fields without x, y dependence untouched and changes none
inside |x|, |y| <= inner. The defect after K snapshots,
    E = ||S_K(T_s u0) - T_s u_K|| / ||T_s u_K - u_K||   on the scored block,
is the finite-s counterpart of the relative transport residual r1 = ||J_K Q0 - Q_K|| / ||Q_K|| (its s -> 0 limit,
computed with the exact tangent as a check): it stays at the floor for a symmetry of the evolution problem at every s
and departs from r1 for a symmetry of the linearised problem only. The size of the transformation is
eps = ||T_s u0 - u0|| / ||u0 - mean u0||; s takes the sign that runs the time part forward (tau > 0 -> s < 0).
Two ways a flow can break down: a tau that depends on x or y makes the term -tau w_t a backward diffusion on part of
the domain for either sign of s (point symmetries of second-order evolution equations have tau = tau(t)), and a flow
that drives u below zero leaves the state space of the porous-medium equation. The script records, per size, the
minimum of T_s u0 and how many flows or evolutions returned non-finite values.

Generators: the algebra (reference); the ten lowest eigenvectors of the polynomial class's pencil (1-6
discovered, 7-10 the near-symmetries next to it); eigenvectors 7-10 with tau removed (well-posed flows: the
positivity mechanism alone); two random fields with tau = 0; with --net, the m directions of a trained exp45
network's span (fields evaluated at the windowed coordinates).

Operators (--op): exact -- the discrete solver; u_t is its right-hand side, the end snapshot the data. fno -- a
learned FNO2d stepper (neural2d/fno2d_pme_s{seed}.pt) with the eigenvectors of its own pencil (neural2d/fno_m6_s{seed}
.npz); u_t is the one-sided fourth-order difference of its rollout, the end snapshot its own rollout, so E measures
how far the learned operator commutes with each finite transformation.

  python exp45_flows.py --seed 42 --out results/neural2d/flows/flows_s42.json
  python exp45_flows.py --seed 42 --only network --net results/neural2d/exact_m8_s42 --out results/neural2d/flows/flows_net_m8_s42.json
  python exp45_flows.py --op fno --device cuda --seed 42 --save neural2d/flows/fno_flows_s42
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch

import exp45_neural_2d as e

NM = len(e.MONS)


def smoothstep(a):
    """C-infinity step: 1 for a <= 0, 0 for a >= 1."""
    a = a.clamp(0.0, 1.0)

    def f(z):
        return torch.where(z > 0, torch.exp(-1.0 / z.clamp_min(1e-12)), torch.zeros_like(z))
    return f(1 - a) / (f(a) + f(1 - a))


def windowed_coordinates(g, inner, outer):
    def chi(z):
        return smoothstep((z.abs() - inner) / (outer - inner))
    return chi(g.X) * g.X, chi(g.Y) * g.Y


class PolyGen:
    """Coefficients C [60] of the polynomial class (component-major, MONS order)."""

    def __init__(self, C):
        self.C = torch.as_tensor(np.asarray(C, dtype=np.float64))

    def fields(self, X, Y, T, U, scale):
        mono = e.monomials(X, Y, T, U)                                                   # [B, 15, n, n]
        C = self.C.to(mono.device)
        return torch.stack([torch.einsum("i,bixy->bxy", C[q * NM:(q + 1) * NM], mono) for q in range(4)], 1)

    def has_tau(self):
        return bool(self.C[2 * NM:3 * NM].abs().max() > 0)


class NetGen:
    """Direction w [m] of a trained exp45 network's span; the network runs in single precision (its rounding, ~1e-7
    relative, is far below the defect of any network direction)."""

    def __init__(self, net, w, score):
        self.net, self.w, self.score = net, torch.as_tensor(np.asarray(w, dtype=np.float64)), score

    def fields(self, X, Y, T, U, scale):
        c = scale["Lx"] / self.score
        u = U * scale["sigma"]
        z = torch.stack([c * X, c * Y, 2 * T - 1, 2 * (u - scale["lo"]) / (scale["hi"] - scale["lo"]) - 1], -1)
        with torch.no_grad():
            o = self.net(z.float()).double()                                             # [B, n, n, m, 4]
        return torch.einsum("bxymc,m->bcxy", o, self.w.to(o.device))

    def has_tau(self):
        return True


class ExactEvolution:
    """The exact discrete solver: S_K by its RK4, u_t by its right-hand side, tangent through its stages."""

    costly_ut = False

    def __init__(self, g, dt, nsub):
        self.g, self.dt, self.nsub, self.op = g, dt, nsub, e.ExactOp(g, dt, nsub)

    def evolve(self, a, K):
        return e.solve(self.g, a, K + 1, self.dt, self.nsub)[:, -1]

    def endpoint(self, d, r, j, K):
        return d.u[r, j + K]

    def ut(self, w):
        return e.rhs(self.g, w)

    def transport(self, u0, d0, K):
        return self.op.transport(u0, d0, K)


class FNOEvolution:
    """A learned FNO2d stepper: S_K by its rollout (also the end snapshot, so E measures commutation with the operator
    itself) and its tangent in double precision; u_t inside the flows by the one-sided fourth-order difference of its
    own rollout in single precision (the operator's generator: time translation then commutes with S_K by
    construction)."""

    costly_ut = True

    def __init__(self, model, dt):
        self.m64, self.m32, self.dt = copy.deepcopy(model).double(), copy.deepcopy(model).float(), dt

    def evolve(self, a, K):
        with torch.no_grad():
            for _ in range(K):
                a = self.m64(a)
        return a

    def endpoint(self, d, r, j, K):
        return self.evolve(d.u[r, j], K)

    def ut(self, w):
        with torch.no_grad():
            s0 = w.float()
            s1 = self.m32(s0)
            s2 = self.m32(s1)
            s3 = self.m32(s2)
            s4 = self.m32(s3)
            return ((-25 * s0 + 48 * s1 - 36 * s2 + 16 * s3 - 3 * s4) / (12 * self.dt)).to(w.dtype)

    def transport(self, u0, d0, K):
        return e.FNOOp(self.m64).transport(u0, d0, K)


class Flow:
    def __init__(self, g, scale, Xw, Yw, evo, project=True):
        self.g, self.s, self.project, self.evo = g, scale, project, evo
        self.X, self.Y = Xw / scale["Lx"], Yw / scale["Lx"]

    def char(self, gen, w, t):
        """P Q[w] and the scaled components o [B, 4, n, n]; w [B, n, n], t [B] (the snapshots' times)."""
        g, sc = self.g, self.s
        v = torch.fft.rfft2(w)
        wx = torch.fft.irfft2(g.ikx * v, s=(g.n, g.n))
        wy = torch.fft.irfft2(g.iky * v, s=(g.n, g.n))
        T = (torch.as_tensor(t, dtype=w.dtype, device=w.device) / sc["T"]).reshape(-1, 1, 1).expand_as(w)
        o = gen.fields(self.X.expand_as(w), self.Y.expand_as(w), T, w / sc["sigma"], sc)
        Q = sc["sigma"] * o[:, 3] - sc["Lx"] * (o[:, 0] * wx + o[:, 1] * wy)
        if gen.has_tau() or not self.evo.costly_ut:
            Q = Q - sc["T"] * o[:, 2] * self.evo.ut(w)
        if self.project:
            Q = torch.fft.irfft2(g.mask * torch.fft.rfft2(Q), s=(g.n, g.n))
        return Q, o

    def steps(self, gen, w, t, s, umax, safety):
        """Per-element RK4 step counts from the transport and diffusion parts of the flow at the initial state."""
        _, o = self.char(gen, w, t)
        xi = torch.maximum(o[:, 0].abs().amax((-2, -1)), o[:, 1].abs().amax((-2, -1))) * self.s["Lx"]
        tau = o[:, 2].abs().amax((-2, -1)) * self.s["T"]
        rate = xi * math.sqrt(self.g.kmax2) + tau * umax * self.g.kmax2
        return torch.ceil(s.abs() * rate * safety).long() + 8

    def apply(self, gen, w, t, s, umax, safety=2.0):
        """RK4 in s for a batch: element b takes ns_b steps of s_b / ns_b and then stands still."""
        ns = self.steps(gen, w, t, s, umax, safety)
        h0 = (s / ns).reshape(-1, 1, 1)
        for i in range(int(ns.max())):
            h = torch.where((i < ns).reshape(-1, 1, 1), h0, torch.zeros_like(h0))
            k1 = self.char(gen, w, t)[0]
            k2 = self.char(gen, w + 0.5 * h * k1, t)[0]
            k3 = self.char(gen, w + 0.5 * h * k2, t)[0]
            k4 = self.char(gen, w + h * k3, t)[0]
            w = w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return w, ns


def snapshots(d, pairs, K, evo=None):
    r, j = pairs[:, 0], pairs[:, 1]
    uK = d.u[r, j + K] if evo is None else evo.endpoint(d, r, j, K)
    return d.u[r, j], uK, d.t[j], d.t[j + K], r


def tau_sign(flow, gen, d, pairs):
    """Sign of s that runs the time part forward (tau > 0 -> s < 0) and tau's share of the field on the block."""
    sl = d.g.sl
    u0, _, t0, _, _ = snapshots(d, pairs, 0)
    o = flow.char(gen, u0, t0)[1][..., sl, sl]
    tau = float(o[:, 2].sum())
    return (-1.0 if tau > 0 else 1.0), float((o[:, 2] ** 2).sum()) / max(float((o ** 2).sum()), 1e-300)


def first_order(flow, d, gen, pairs, K):
    """r1 = ||J_K Q0 - Q_K|| / ||Q_K|| on the scored block (the operator's tangent), summed squares over pairs."""
    sl = d.g.sl
    u0, uK, t0, tK, _ = snapshots(d, pairs, K, flow.evo)
    Q0 = flow.char(gen, u0, t0)[0]
    QK = flow.char(gen, uK, tK)[0]
    JQ = flow.evo.transport(u0, Q0[:, None], K)[:, -1, 0]
    return math.sqrt(float(((JQ - QK)[..., sl, sl] ** 2).sum()) / float((QK[..., sl, sl] ** 2).sum()))


def defects(flow, d, gen, pairs, K, eps_list, sign, safety=2.0):
    """E(eps) aggregated over pairs (ratio of summed squares; non-finite if any pair fails), the realised eps, the
    flows' largest step count, the minimum of the transformed initial snapshots (full grid, scored block) and the
    number of pairs whose flow or evolution failed (non-finite)."""
    sl = d.g.sl
    u0, uK, t0, tK, r = snapshots(d, pairs, K, flow.evo)
    P = len(pairs)
    Q0 = flow.char(gen, u0, t0)[0]
    base = ((u0 - u0.mean((-2, -1), keepdim=True))[..., sl, sl] ** 2).sum((-2, -1)).sqrt()
    q0 = (Q0[..., sl, sl] ** 2).sum((-2, -1)).sqrt()
    umax = d.u[r].amax((1, 2, 3)) * 1.5
    W, T, UM = torch.cat([u0, uK]), torch.cat([t0, tK]), torch.cat([umax, umax])
    out = {}
    for x in eps_list:
        s = sign * x * base / q0.clamp_min(1e-30)
        ab, ns = flow.apply(gen, W, T, torch.cat([s, s]), UM, safety)
        a, b = ab[:P], ab[P:]
        flow_ok = torch.isfinite(a).all((-2, -1)) & torch.isfinite(b).all((-2, -1))
        sk = flow.evo.evolve(torch.where(flow_ok.reshape(-1, 1, 1), a, u0), K)
        G = (sk - b)[..., sl, sl]
        Dl = (b - uK)[..., sl, sl]
        solve_ok = flow_ok & torch.isfinite(G).all((-2, -1))
        n_flow, n_solve = int((~flow_ok).sum()), int((flow_ok & ~solve_ok).sum())
        ok = solve_ok
        if bool(flow_ok.any()):
            umin = [float(a[flow_ok].min()), float(a[flow_ok][..., sl, sl].min())]
        else:
            umin = [float("nan"), float("nan")]
        num, den = float((G[ok] ** 2).sum()), float((Dl[ok] ** 2).sum())
        real = (((a - u0)[..., sl, sl] ** 2).sum((-2, -1)).sqrt() / base)[ok]
        E = math.sqrt(num / den) if (n_flow + n_solve == 0 and den > 0) else float("inf")
        out[str(x)] = dict(E=E, eps=float(real.mean()) if len(real) else float("nan"), steps=int(ns.max()), umin=umin,
                           flow_fail=n_flow, solve_fail=n_solve)
    return out


def load_net(stem):
    ck = torch.load(stem + ".pt", weights_only=False)
    net = e.GenNet(ck["m"], ck["width"], ck["head"])
    net.load_state_dict(ck["state"])
    return net.float().eval(), np.asarray(ck["W"]), ck["scale"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="exact", choices=["exact", "fno"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", default="0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--starts", default="2,18")
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--inner", type=float, default=6.0)
    ap.add_argument("--outer", type=float, default=7.75)
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--safety", type=float, default=2.0, help="RK4 steps in s per unit of |s| x (transport + diffusion rate)")
    ap.add_argument("--only", default=None, help="comma-separated generator names to run (prefix match)")
    ap.add_argument("--net", default=None, help="stem of a trained exp45 run (.pt) whose span directions are added")
    ap.add_argument("--poly", default=None, help="exp45 .npz whose polynomial eigenvectors to use (fno default: "
                                                 "neural2d/fno_m6_s{seed}.npz under FINS_RESULTS_DIR)")
    ap.add_argument("--fno-tag", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--save", default=None, help="output stem under FINS_RESULTS_DIR (json)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = e.Grid()
    t0 = time.time()
    dtr = e.Data(g, e.trajectories(g, 16, dseed, 51, 0.02, 3), 0.02)
    dev = e.Data(g, e.trajectories(g, args.n_eval, eseed, 51, 0.02, 3), 0.02, dtr.s)
    if args.op == "exact" and args.poly is None:
        op = e.ExactOp(g, 0.02, 3)
        N, DA, G0, trBB = e.pencil(op, dtr, e.chars_poly, e.pairs_of(16, 51, args.K, 4), args.K, 2)
        lam, C = e.gsolve(N, DA + trBB / np.trace(G0) * G0)
    else:
        path = args.poly or os.path.join(e.RESULTS, "neural2d", f"fno_m6_s{s}{args.fno_tag}.npz")
        z = np.load(path)
        lam, C = z["poly_lam"], z["poly_C"]
    gd = g.cast(torch.float64, args.device)
    dev = dev.cast(torch.float64, args.device)
    if args.op == "exact":
        evo = ExactEvolution(gd, 0.02, 3)
    else:
        fa = argparse.Namespace(seed=s, fno_tag=args.fno_tag, train_fno_if_missing=False)
        model = e.load_fno(fa, g.cast(torch.float32)).to(args.device)
        evo = FNOEvolution(model, 0.02)
    Ct = e.true_coefficients()
    Xw, Yw = windowed_coordinates(gd, args.inner, args.outer)
    flow = Flow(gd, dtr.s, Xw, Yw, evo, project=not args.no_project)
    eps_list = [float(x) for x in args.eps.split(",")]
    pairs = torch.tensor([(r, j) for r in range(args.n_eval) for j in (int(v) for v in args.starts.split(","))])
    rng = np.random.default_rng(s)
    gens = [(f"true {n}", "true", PolyGen(Ct[:, i])) for i, n in enumerate(e.NAMES)]
    gens += [(f"eigenvector {i + 1}", "discovered" if i < 6 else "near", PolyGen(C[:, i] / np.linalg.norm(C[:, i])))
             for i in range(10)]
    for i in range(6, 10):
        c = C[:, i] / np.linalg.norm(C[:, i])
        c[2 * NM:3 * NM] = 0.0
        gens.append((f"eigenvector {i + 1} without tau", "near, tau removed", PolyGen(c)))
    for j in range(2):
        c = rng.standard_normal(4 * NM)
        c[2 * NM:3 * NM] = 0.0
        gens.append((f"random {j + 1}", "random", PolyGen(c)))
    if args.net:
        stem = args.net if os.path.exists(args.net + ".pt") else os.path.join(e.RESULTS, args.net)
        net, W, nscale = load_net(stem)
        net = net.to(args.device)
        for j in range(W.shape[1]):
            gens.append((f"network direction {j + 1}", "network", NetGen(net, W[:, j], g.score)))
    if args.only:
        keep = args.only.split(",")
        gens = [x for x in gens if any(x[0].startswith(k) for k in keep)]
    report = dict(op=args.op, seed=s, data_seed=dseed, eval_seed=eseed, K=args.K, inner=args.inner, outer=args.outer,
                  project=not args.no_project, safety=args.safety, net=args.net, n_eval=args.n_eval,
                  lam=np.asarray(lam[:12]).tolist(), gens={})
    print(f"[flows, {args.op}] seed {s}: data {dseed}/{eseed}, K {args.K}, eps {eps_list}, {len(pairs)} pairs, "
          f"{args.device}; windowed coordinates |x|, |y| <= {args.inner:g} kept, zero beyond {args.outer:g}; projection "
          f"{'on' if flow.project else 'off'}; polynomial lowest " + " ".join(f"{v:.1e}" for v in lam[:10])
          + f" ({time.time() - t0:.0f} s)", flush=True)
    out = args.out or (os.path.join(e.RESULTS, args.save + ".json") if args.save else None)
    for name, kind, gen in gens:
        t1 = time.time()
        sign, tshare = tau_sign(flow, gen, dev, pairs)
        r1 = first_order(flow, dev, gen, pairs, args.K)
        res = defects(flow, dev, gen, pairs, args.K, eps_list, sign, args.safety)
        report["gens"][name] = dict(kind=kind, sign=sign, tau_share=tshare, r1=r1, E=res)
        print(f"  {name:<22s} tau share {tshare:.2f}, r1 {r1:.2e} | E(eps): "
              + "  ".join(f"{res[str(x)]['E']:.2e} ({res[str(x)]['eps']:.3f})" for x in eps_list)
              + f"  [{time.time() - t1:.0f} s, max steps {max(v['steps'] for v in res.values())}]", flush=True)
        bad = [x for x in eps_list if not res[str(x)]["E"] < 10 * res[str(eps_list[0])]["E"]]
        if bad:
            x = bad[0]
            v = res[str(x)]
            print(f"      first eps with E >= 10 E({eps_list[0]:g}) or non-finite: {x:g}; min of T_s u0 {v['umin'][0]:+.3f} "
                  f"(block {v['umin'][1]:+.3f}); failed flows {v['flow_fail']}, failed evolutions {v['solve_fail']} of "
                  f"{len(pairs)}", flush=True)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump(report, f, indent=1)
    if out:
        print(f"saved {out}")


if __name__ == "__main__":
    main()
