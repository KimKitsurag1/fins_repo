
## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 4 (4.0) | 0.01 +/- 0.00 | 0.0 | 0.01 +/- 0.00 | 0.10 + 0.00 | 4.0 / 4 | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 |
| null: uniform on the class (4 of 30) | 3 | 4 (4.0) | 3.46 +/- 0.01 | 0.0 | 3.46 +/- 0.01 | 2.54 + 0.00 | 0.0 / 4 | 0.23 | 0.35 | 0.35 | 0.34 | 0.35 |
| null: Gaussian coefficients (4 of 30) | 3 | 4 (4.0) | 2.85 +/- 0.03 | 0.0 | 2.85 +/- 0.03 | 2.42 + 0.00 | 0.0 / 4 | 0.31 | 0.14 | 0.80 | 0.25 | 0.35 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 708.0x, 533.4x, 495.2x; lambda_(k+1) / lambda_k 812.6, 381.3, 496.4; rank from the largest gap 4, 4, 4; relative change 0.44, 0.52, 0.48; band check u_t share 1.5e-16, 2.0e-15, 3.9e-16
