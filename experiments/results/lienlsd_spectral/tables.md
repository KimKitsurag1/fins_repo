
## burgers: 3 seeds, ground truth finite 5; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (5)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_RHS_spec (uses the equation) | 3 | 5 (5.0) | 0.46 +/- 0.33 | 0.0 | 0.46 +/- 0.33 | 0.67 + 0.00 | 3.7 / 5 | 0.85 | 0.95 | 0.90 | 0.96 | 0.72 | 0.95 |
| LieNLSD_RHS_spec, own count (uses the equation) | 3 | 3-5 (4.0) | 1.26 +/- 0.53 | 1.0 | 0.26 +/- 0.37 | 0.36 + 2.47 | 1.7 / 5 | 0.89 | 0.89 | 0.76 | 0.77 | 0.33 | 0.61 |
| LieNLSD_F_spec | 3 | 5 (5.0) | 0.70 +/- 0.23 | 0.0 | 0.70 +/- 0.23 | 0.99 + 0.00 | 2.7 / 5 | 0.79 | 0.92 | 0.84 | 0.96 | 0.64 | 0.94 |
| LieNLSD_F_spec, own count | 3 | 2-5 (3.3) | 1.85 +/- 1.01 | 1.7 | 0.18 +/- 0.25 | 0.30 + 4.11 | 1.0 / 5 | 0.89 | 0.89 | 0.78 | 0.61 | 0.34 | 0.51 |


## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_RHS_spec (uses the equation) | 3 | 4 (4.0) | 0.64 +/- 0.45 | 0.0 | 0.64 +/- 0.45 | 0.91 + 0.00 | 2.0 / 4 | 0.83 | 0.97 | 0.88 | 0.74 | 0.47 |
| LieNLSD_RHS_spec, own count (uses the equation) | 3 | 3-4 (3.3) | 0.67 +/- 0.47 | 0.7 | 0.00 +/- 0.00 | 0.00 + 1.64 | 2.0 / 4 | 0.83 | 0.97 | 0.87 | 0.72 | 0.33 |
| LieNLSD_F_spec | 3 | 4 (4.0) | 0.95 +/- 0.07 | 0.0 | 0.95 +/- 0.07 | 1.37 + 0.00 | 1.0 / 4 | 0.83 | 0.95 | 0.84 | 0.46 | 0.33 |
| LieNLSD_F_spec, own count | 3 | 2-4 (2.7) | 1.62 +/- 0.54 | 1.3 | 0.28 +/- 0.40 | 0.39 + 3.29 | 1.0 / 4 | 0.94 | 0.95 | 0.84 | 0.13 | 0.22 |


## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_RHS_spec (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 4.00 +/- 0.00 | 0.00 + 9.87 | 6.0 / 6 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD_RHS_spec, own count (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 4.00 +/- 0.00 | 0.00 + 9.87 | 6.0 / 6 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD_F_spec | 3 | 10 (10.0) | 0.09 +/- 0.05 | 0.0 | 4.09 +/- 0.05 | 0.29 + 9.87 | 6.0 / 6 | 0.68 | 0.99 | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 |
| LieNLSD_F_spec, own count | 3 | 9-10 (9.7) | 0.24 +/- 0.17 | 0.0 | 3.91 +/- 0.30 | 0.49 + 9.05 | 5.0 / 6 | 0.68 | 0.99 | 0.97 | 0.93 | 0.93 | 0.97 | 0.93 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_RHS_spec (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 0.00 +/- 0.00 | 0.00 + 0.00 | 10.0 / 10 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD_RHS_spec, own count (uses the equation) | 3 | 10 (10.0) | 0.00 +/- 0.00 | 0.0 | 0.00 +/- 0.00 | 0.00 + 0.00 | 10.0 / 10 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD_F_spec | 3 | 10 (10.0) | 0.13 +/- 0.06 | 0.0 | 0.13 +/- 0.06 | 0.35 + 0.00 | 10.0 / 10 | 0.68 | 0.99 | 0.98 | 0.97 | 0.97 | 0.98 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 |
| LieNLSD_F_spec, own count | 3 | 9-10 (9.7) | 0.45 +/- 0.40 | 0.3 | 0.11 +/- 0.08 | 0.31 + 0.82 | 9.0 / 10 | 0.68 | 0.99 | 0.97 | 0.79 | 0.79 | 0.98 | 0.81 | 1.00 | 0.99 | 0.98 | 0.97 |

