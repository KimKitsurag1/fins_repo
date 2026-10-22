
## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 4 (4.0) | 2.20 +/- 0.01 | 0.0 | 2.20 +/- 0.01 | 2.24 + 0.00 | 1.7 / 4 | 0.33 | 0.30 | 0.53 | 0.93 | 0.90 |
| null: uniform on the class (4 of 30) | 3 | 4 (4.0) | 3.46 +/- 0.01 | 0.0 | 3.46 +/- 0.01 | 2.53 + 0.00 | 0.0 / 4 | 0.23 | 0.35 | 0.35 | 0.35 | 0.35 |
| null: Gaussian coefficients (4 of 30) | 3 | 4 (4.0) | 2.85 +/- 0.03 | 0.0 | 2.85 +/- 0.03 | 2.42 + 0.00 | 0.0 / 4 | 0.31 | 0.14 | 0.80 | 0.25 | 0.35 |

FINS: all below lambda_(k+1) on 0/3 seeds; worst margin 0.2x, 0.2x, 0.2x; lambda_(k+1) / lambda_k 2.4, 1.9, 1.6; rank from the largest gap 4, 3, 1; relative change 0.76, 0.74, 0.73; band check u_t share 2.4e-01, 2.3e-01, 2.3e-01
