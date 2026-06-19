# FINS

FINS discovers infinitesimal Lie point symmetries of evolutionary PDEs
from trajectory data alone, by enforcing a Fréchet-invariance condition
on a learned solution operator:
    dS_θ[u_0](Q_0) = Q(x, t)
for each candidate symmetry's characteristic Q. Sequential Gram–Schmidt
deflation produces an orthonormal basis of generators; the algebra's
rank is detected automatically through a spectral gap in generator
norms. The same Fréchet-invariance principle extends to natural images
through a pretrained encoder surrogate.

This repository contains two independent experiment tracks:
  - `pde/` — PDE-side symmetry discovery on heat, Burgers, and KdV
  - `vision/` — image-side experiment on Pet and CIFAR-10
## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. CUDA 12.x recommended for the PDE side
(training the solution operator + sequential deflation is GPU-bound);
the vision side runs on a single A100/T4 in 30 min – 1.5 h per
seed depending on the GPU.


## Repository layout

```
.
├── README.md
├── requirements.txt
├── pde/                          PDE-side experiments
│   ├── config.py                 ExperimentConfig dataclass
│   ├── data_generation.py        PDE solvers (heat, Burgers, KdV)
│   ├── shared_modules.py         FNO / generator-FNO building blocks
│   ├── shared_modules_v2.py      FNOSurrogateEquivariant (S_θ for FINS)
│   ├── ground_truth.py           analytical Lie algebras + metrics
│   ├── method_fins.py            *** FINS main algorithm ***
│   ├── method_fins_components.py building blocks (deflation, FD utils)
│   ├── method_augerino.py        Augerino baseline
│   ├── method_liegan.py          LieGAN baseline
│   ├── method_laligan.py         LaLiGAN baseline
│   ├── method_lig.py             LIG baseline (equation-aware)
│   ├── method_lienlsd.py         LieNLSD baseline (equation-aware)
│   └── run_single_pde.py         end-to-end runner: trains S_θ, runs
│                                 FINS + 5 baselines, prints metrics
└── vision/                       Image-side experiments
    ├── fins_cifar.py             trains FINS sym_gen + classifier on
    │                             CIFAR-10, evaluates corruptions (v7 method)
    ├── fins_pet.py               trains six-way breed classifier on
    │                             Oxford-IIIT Pet, evaluates corruptions
    │                             (v5/v6 method — see Section 6 of paper)
    └── visualize_cifar_generators.py
                                  per-generator visualization from a
                                  trained sym_gen checkpoint
```


## Reproducing the paper

### PDE Table

```bash
cd pde
python run_single_pde.py --pde heat    --seed 42
python run_single_pde.py --pde burgers --seed 42
python run_single_pde.py --pde kdv     --seed 42
```

Each invocation:
1. Generates training and test data via spectral solvers
2. Trains the solution operator S_θ (`train_solution_operator`)
3. Discovers FINS generators via sequential deflation (`train_fins`)
4. Trains and evaluates the 5 baselines (LieGAN, LaLiGAN, Augerino, LIG, LieNLSD)
5. Prints a metrics table (rank-aware Grassmann distance d_G, mean
   cosine, algebra closure error ACE, estimated rank r̂) and writes
   per-method metrics to disk

Total runtime: ~6–8 GPU-hours per (PDE, seed). Use `--quick` for a
sanity-check run (~30 min) with reduced grid/training budgets.

For the variance bars in the paper, repeat with seeds 42, 52, 152 and
aggregate across runs (mean ± std).

### Vision Tables

The vision section uses two scripts, one per dataset:

```bash
cd vision
python fins_cifar.py    # CIFAR-10 (v7 method, ~30-45 min/seed on A100)
python fins_pet.py      # Oxford-IIIT Pet (v5/v6 method, ~25 min/seed on A100)
```

Each script trains FINS sym_gen + downstream classifier and compares
against four augmentation baselines (NoAug, RandAug, AutoAug,
TrivialAug). The CIFAR script uses the upgraded v7 FINS method
(combined L_inv + L_τ loss, Fréchet-invariance against a frozen DINOv2
encoder), described in Section 6 of the paper. The Pet script uses an
earlier v5/v6 variant of the FINS method (Olver-style loss on a
trained classifier surrogate); this difference is documented in
Section 6 footnote.

The CONFIG dicts at the top of each script control dataset paths,
number of seeds, and augmentation parameters. Outputs:

- `outputs/fins_cifar_results.json` and `outputs/fins_pet_results.json`
  — per-seed accuracy across methods on clean and 9 corruption types
- `outputs/sym_gen_checkpoints/*.pt` (CIFAR) and
  `outputs/sym_gen_pet_checkpoints/*.pt` (Pet) — sym_gen weights per seed

Auto-resumes from existing checkpoints; safe to re-run after interruption.

### Per-generator visualization

```bash
cd vision
python visualize_cifar_generators.py
```

Loads the most recent sym_gen checkpoint from
`outputs/sym_gen_checkpoints/`, generates a panel per CIFAR-10 test
image showing the action of each of the six discovered generators
(ξ, η components rendered as vector fields; φ as RGB difference;
full integrated flow at τ=0.8). Also produces an augmented gallery
illustrating what the downstream classifier sees as training data.


## Notes on the PDE pipeline

- `S_θ` is trained once per (PDE, seed) and cached to disk
  (`S_theta_*.pt`). Subsequent runs reload it.
- The default solution-operator backend is the FNO with architectural
  translation-equivariance (`FNOSurrogateEquivariant` from
  `shared_modules_v2.py`), obtained by removing the explicit x-channel
  from the FNO input. This is the architecture used to produce the
  numbers in the paper. The fallback `FNOSurrogate` from
  `shared_modules.py` is also available via
  `train_solution_operator(equivariant=False)`, but loses architectural
  ∂_x equivariance and produces lower symmetry-recovery quality.
- The optional CNO backend in `run_single_pde.py --operator_backend cno`
  expects a separate `shared_operators.py` module providing a
  `LocalGeneratorNO` class with the same API as `LocalGeneratorFNO`.
  This is used only for an optional diagnostic time-derivative
  surrogate N_θ[u] ≈ u_t; FINS itself uses S_θ and is unaffected by
  this choice.


## License

Code released for the purpose of double/single-blind peer review.
A permissive open-source license will be applied at publication time.
