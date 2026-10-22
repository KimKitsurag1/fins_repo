"""Robustness of the superclass test (G4/G5) of the confirmatory run: reliability covariates, raw distance, leave one
superclass out, tree breakdown, sub-pools, subspace dimension, relative pencil.

  python semantic_robustness.py confirm_resnet50_v1.npz confirm_convnext_tiny.npz
"""
import json
import sys
import numpy as np
import fins_semantic as fs

def halves_D(G, pool, I, nG, k, ref, splits=10, seed=0, metric="contrastive", Mden=None):
    rng = np.random.RandomState(seed)
    Nall = sum(fs.class_pencil(G, pool, I, g) for g in ref) / len(ref) if metric == "contrastive" else Mden
    D = {}
    for _ in range(splits):
        p = rng.permutation(pool)
        V = []
        for h in (p[:len(p) // 2], p[len(p) // 2:]):
            Ns = [fs.class_pencil(G, h, I, g) for g in range(nG)]
            if metric == "contrastive":
                Nbar = sum(Ns[g] for g in ref) / len(ref)
                V.append([fs.orthonormal(fs.solve(Ns[g], Nbar)[1][:, :k], Nall) for g in range(nG)])
            else:
                V.append([fs.lowest(Ns[g], Mden, k)[1] for g in range(nG)])
        for a in range(nG):
            for b in range(nG):
                D.setdefault((a, b), []).append(fs.missing(V[0][a], V[1][b], Nall))
    nG_ = nG
    raw = np.array([[0.5 * (np.mean(D[(a, b)]) + np.mean(D[(b, a)])) for b in range(nG_)] for a in range(nG_)])
    own = np.array([np.mean(D[(a, a)]) for a in range(nG_)])
    spec = raw - 0.5 * (own[:, None] + own[None, :])
    return raw, spec, own

def resid_test(S, covs, labels):
    iu = np.triu_indices(len(labels), 1)
    A = np.c_[np.ones(len(iu[0])), *[c[iu] for c in covs]]
    R = np.zeros_like(S)
    R[iu] = S[iu] - A @ np.linalg.lstsq(A, S[iu], rcond=None)[0]
    return fs.superclass_test(R, labels)

animal = {"domestic_cat", "dog", "big_cat", "bird"}
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    net = json.loads(str(z["meta"]))["network"]
    names = [str(x) for x in z["group_names"]]
    nG, G, gi = len(names), z["G"], z["image_group"]
    I = list(range(G.shape[1]))
    ref = [g for g in range(nG) if (gi == g).sum() > 0]
    ext = [i for i, n in enumerate(names) if ":" in n]
    labels = np.array([names[i].split(":")[0] for i in ext])
    Rd = fs.readout_distance(z["readout_mean"])[np.ix_(ext, ext)]
    print(f"\n===== {net}")
    raw, spec, own = halves_D(G, np.arange(len(gi)), I, nG, 8, ref)
    S, Wr, Oe = spec[np.ix_(ext, ext)], raw[np.ix_(ext, ext)], own[ext]
    omean, odiff = 0.5 * (Oe[:, None] + Oe[None, :]), np.abs(Oe[:, None] - Oe[None, :])
    print("  G4 (repeat) specificity: %+.3f p %.4f" % fs.superclass_test(S, labels))
    print("  specificity, regressed on readout + pair reliability (mean, |diff|): %+.3f p %.4f" % resid_test(S, [Rd, omean, odiff], labels))
    print("  raw distance: %+.3f p %.4f" % fs.superclass_test(Wr, labels))
    print("  raw distance, regressed on readout + pair reliability: %+.3f p %.4f" % resid_test(Wr, [Rd, omean, odiff], labels))
    print("  own split-half by superclass: " + "  ".join(f"{s} {Oe[labels == s].mean():.2f}" for s in dict.fromkeys(labels)))
    # tree breakdown
    iu = np.triu_indices(len(ext), 1)
    a_, b_ = labels[iu[0]], labels[iu[1]]
    same = a_ == b_
    aa = (~same) & np.isin(a_, list(animal)) & np.isin(b_, list(animal))
    ff = (~same) & ~np.isin(a_, list(animal)) & ~np.isin(b_, list(animal))
    af = (~same) & ~aa & ~ff
    print(f"  mean specificity: within superclass {S[iu][same].mean():+.3f} (36), animal-animal {S[iu][aa].mean():+.3f} ({aa.sum()}), "
          f"artefact-artefact {S[iu][ff].mean():+.3f} ({ff.sum()}), animal-artefact {S[iu][af].mean():+.3f} ({af.sum()})")
    # leave one superclass out
    loo = []
    for s in dict.fromkeys(labels):
        keep = labels != s
        d, p = fs.superclass_test(S[np.ix_(keep, keep)], labels[keep], n_perm=5000)
        loo.append(f"{s} {d:+.2f} (p {p:.4f})")
    print("  leave one superclass out: " + "  ".join(loo))
    # per superclass: its within mean against its pairs with the others
    per = []
    for s in dict.fromkeys(labels):
        w = S[iu][(a_ == s) & (b_ == s)].mean()
        bt = S[iu][((a_ == s) | (b_ == s)) & ~same].mean()
        per.append(f"{s} {w:+.2f}/{bt:+.2f}")
    print("  per superclass, within / with the others: " + "  ".join(per))
    # sub-pools
    for label, pool in (("Pet images", np.flatnonzero(np.isin(gi, [0, 1]))), ("Imagenette images", np.flatnonzero(gi >= 2)),
                        ("dog images", np.flatnonzero(gi == 1)), ("tench images", np.flatnonzero(gi == names.index("tench")))):
        _, sp, _ = halves_D(G, pool, I, nG, 8, ref)
        print(f"  pool {label} ({len(pool)}): G4 %+.3f p %.4f" % fs.superclass_test(sp[np.ix_(ext, ext)], labels))
    for k in (4, 12):
        _, sp, _ = halves_D(G, np.arange(len(gi)), I, nG, k, ref)
        print(f"  k = {k}: G4 %+.3f p %.4f" % fs.superclass_test(sp[np.ix_(ext, ext)], labels))
    Mden = z["M"].sum(0) / z["count"].sum()
    _, sp, _ = halves_D(G, np.arange(len(gi)), I, nG, 8, ref, metric="relative", Mden=Mden)
    print("  relative pencil (feature metric), same images: G4 %+.3f p %.4f" % fs.superclass_test(sp[np.ix_(ext, ext)], labels))
