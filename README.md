# FINS

Code for the paper *Equation-Free Discovery of Lie Point Symmetries from Learned
Solution Operators* (I. Markov, A. Hvatov), submitted to *Machine Learning and
Knowledge Extraction* (MDPI).

FINS (Fréchet-Invariant Neural Symmetry Discovery) finds the Lie point symmetries
of an evolution equation u_t = F[u] from a solution operator, exact or learned,
without the equation and without its order. A symmetry maps solutions to
solutions, so the Jacobian of the solution operator transports its characteristic
Q = φ − ξ u_x − η u_t. Over a polynomial class of generators the transport
residual is a ratio of quadratic forms, and discovery is one generalised eigenvalue
problem N c = λ D c; the eigenvectors with the smallest eigenvalues are the
discovered generators. A weak form moves every derivative of the data onto smooth
test functions.

## Layout

```
experiments/                 everything behind the paper
  PROTOCOL.md                evaluation protocol, fixed before the final runs
  README_EXPERIMENTS.md      log of all runs, development runs included
  exp_common.py              exact solvers, Jacobian-vector products, shared utilities
  exp39_localised.py         FINS, pointwise, through periodic steppers (KdV)
  exp40_window.py            FINS, pointwise, on four-period windows (Burgers, heat)
  exp49_weak.py              FINS, weak form
  exp41_baselines.py         LieGAN, LaLiGAN, Augerino, LIG, LieNLSD (Burgers, heat, KdV)
  exp54_baselines_new.py     the same baselines on Benjamin-Ono and Kuramoto-Sivashinsky
  exp50_two_stage.py         identify-then-Lie pipeline (weak SINDy, then Lie's algorithm)
  exp56_wsindy.py            weak SINDy in the authors' reference configuration
  exp71_kreider.py           the model-free jet-space method of Kreider et al.
  kreider_check.py           check of the reimplementation against the authors' code
  kreider_reproduce.py       the authors' convergence examples
  exp51_noise_weak.py        noise study (heat, KdV)
  exp52_bo.py                Benjamin-Ono
  exp53_ks.py                Kuramoto-Sivashinsky
  exp55_extra.py             two-stage pipeline on Burgers and with one library for
                             every equation; weak FINS through the learned heat operators
  exp57_fburgers.py          fractional Burgers family
  exp72_utility.py           discovered generators as data augmentation
  exp42_tables.py            scoring and chance levels of the main comparison
  exponent_readout.py        time-scaling exponent read from a span
  exp56_tables.py            paired readouts (Table A2)
  tables_0710.py             rows of the noise table
  exp71_rows.py              rows of Kreider et al.
  exp72_table.py             augmentation table and the numbers of its section
  run_all.py                 CPU jobs of the noise, Benjamin-Ono and KS studies
  run_fresh.py               check on unseen evaluation data (seeds 2042-2044)
  colab/                     Colab notebooks of the GPU runs, named as in the log
  FINS_experiments_colab.ipynb   Colab runner of the early experiments (exp1-exp48)
  paper_figures/             scripts of Figures 1-7
  results/                   summaries behind every table (about 25 MB)
pde/                         solvers, operators, baselines, ground-truth algebras;
                             method_kreider.py; run_single_pde.py and method_fins.py
                             are the earlier deflation-based version of FINS
vision/                      image experiments of the earlier conference version;
                             not used in this paper
```

The other `exp*.py` scripts are development and side experiments (two-dimensional
equations, hidden variables, diagnostics). Each is described in its docstring and
in `README_EXPERIMENTS.md`; none of them produces a number in the paper.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Everything runs on a CPU; a GPU only shortens the training
of the learned operators. The baselines need extra packages, listed as optional in
`requirements.txt`: `torchdiffeq` (LIG), `pysindy==2.1.0` (the sensitivity
configuration of the two-stage pipeline) and Adan (LieNLSD's optimiser,
`pip install git+https://github.com/sail-sg/Adan.git`).

## Reproducing the paper

Tables from the stored summaries in `experiments/results` (noise table, rows of
Kreider et al., augmentation table, paired readouts):

```bash
cd experiments
python tables_0710.py --out tab_noise.tex
python exp71_rows.py
python exp72_table.py
python exp56_tables.py paired
```

Figures:

```bash
cd experiments/paper_figures
python make_fig_comparison.py
python make_fig_noise.py
python make_fig_fburgers.py
python make_fig_augment.py
```

The main comparison (`exp42_tables.py`) and Figures 1, 2 and 5
(`make_fig_method.py`, `make_fig_data.py`, `make_fig_spectra.py`) read individual
runs, which are too large for the repository (about 16 GB). Set `FINS_RESULTS_DIR`
to the directory of a rerun.

Rerunning the studies: every study script prints its command line in its
docstring (`python <script> --help`). The protocol runs (Burgers, heat, KdV) are
specified in `PROTOCOL.md`; `run_fresh.py` lists their commands for the learned
operators. `run_all.py` runs the CPU jobs of the noise, Benjamin-Ono and
Kuramoto-Sivashinsky studies (one worker per core); the GPU runs are the notebooks
in `experiments/colab/`. Seeds are 42-44 for the
learned operators and baselines and 42-51 for the exact solvers and the noise
study; evaluation trajectories use data seed 1000 + s.

## Experiment log

`experiments/README_EXPERIMENTS.md` records every run in order, with settings and
expected outcomes written before the runs on the reported seeds. Its terms differ
in places from the paper's; the glossary at its top maps them.

## License

MIT, see `LICENSE`.
