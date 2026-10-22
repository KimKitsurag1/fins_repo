
## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 4 (4.0) | 0.01 +/- 0.00 | 0.0 | 0.01 +/- 0.00 | 0.12 + 0.00 | 4.0 / 4 | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 |
| null: uniform on the class (4 of 30) | 3 | 4 (4.0) | 3.47 +/- 0.01 | 0.0 | 3.47 +/- 0.01 | 2.54 + 0.00 | 0.0 / 4 | 0.23 | 0.34 | 0.35 | 0.34 | 0.35 |
| null: Gaussian coefficients (4 of 30) | 3 | 4 (4.0) | 2.85 +/- 0.03 | 0.0 | 2.85 +/- 0.03 | 2.42 + 0.00 | 0.0 / 4 | 0.31 | 0.14 | 0.80 | 0.26 | 0.36 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 630.0x, 358.0x, 377.2x; lambda_(k+1) / lambda_k 699.5, 280.3, 344.4; rank from the largest gap 4, 4, 4; relative change 0.51, 0.60, 0.55; band check u_t share 2.8e-16, 4.9e-15, 8.4e-16
