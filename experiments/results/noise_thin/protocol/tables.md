
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS, exact operator (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 4.00 +/- 0.00 | 0.00 + 9.87 | 6.0 / 6 | 0.57 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 4.99 +/- 0.01 | 0.0 | 8.99 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.41 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.72 | 0.60 | 0.39 | 0.37 | 0.55 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS, exact operator (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 0.00 +/- 0.00 | 0.00 + 0.00 | 10.0 / 10 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.32 +/- 0.02 | 0.0 | 8.32 +/- 0.02 | 3.87 + 0.00 | 0.0 / 10 | 0.19 | 0.40 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.36 +/- 0.12 | 0.0 | 6.36 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.71 | 0.60 | 0.37 | 0.36 | 0.53 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS, exact operator: all below lambda_(k+1) on 3/3 seeds; worst margin 30404.2x, 25671.9x, 6833.3x; lambda_(k+1) / lambda_k 26806.3, 29517.4, 8049.0; rank from the largest gap 6, 6, 6; relative change 0.12, 0.22, 0.26; band check u_t share 1.1e-27, 4.6e-28, 9.2e-28

## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS, exact operator (uses the equation) | 3 | 4 (4.0) | 0.01 +/- 0.00 | 0.0 | 0.01 +/- 0.00 | 0.10 + 0.00 | 4.0 / 4 | 0.89 | 1.00 | 1.00 | 1.00 | 0.99 |
| null: uniform on the class (4 of 30) | 3 | 4 (4.0) | 3.46 +/- 0.01 | 0.0 | 3.46 +/- 0.01 | 2.53 + 0.00 | 0.0 / 4 | 0.23 | 0.35 | 0.35 | 0.35 | 0.35 |
| null: Gaussian coefficients (4 of 30) | 3 | 4 (4.0) | 2.85 +/- 0.03 | 0.0 | 2.85 +/- 0.03 | 2.42 + 0.00 | 0.0 / 4 | 0.31 | 0.14 | 0.80 | 0.25 | 0.35 |

FINS, exact operator: all below lambda_(k+1) on 3/3 seeds; worst margin 838.5x, 568.5x, 519.1x; lambda_(k+1) / lambda_k 1183.8, 650.4, 632.5; rank from the largest gap 2, 2, 2; relative change 0.39, 0.46, 0.42; band check u_t share 1.6e-16, 2.1e-15, 4.1e-16
