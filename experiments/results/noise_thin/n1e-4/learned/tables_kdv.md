
## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 4 (4.0) | 0.07 +/- 0.02 | 0.0 | 0.07 +/- 0.02 | 0.27 + 0.00 | 4.0 / 4 | 0.76 | 0.99 | 0.98 | 1.00 | 1.00 |
| null: uniform on the class (4 of 30) | 3 | 4 (4.0) | 3.46 +/- 0.01 | 0.0 | 3.46 +/- 0.01 | 2.53 + 0.00 | 0.0 / 4 | 0.23 | 0.35 | 0.35 | 0.35 | 0.35 |
| null: Gaussian coefficients (4 of 30) | 3 | 4 (4.0) | 2.85 +/- 0.03 | 0.0 | 2.85 +/- 0.03 | 2.42 + 0.00 | 0.0 / 4 | 0.31 | 0.14 | 0.80 | 0.25 | 0.35 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 62.4x, 71.8x, 61.0x; lambda_(k+1) / lambda_k 9.0, 11.7, 8.8; rank from the largest gap 3, 4, 3; relative change 0.40, 0.46, 0.43; band check u_t share 4.8e-05, 4.2e-05, 4.2e-05
