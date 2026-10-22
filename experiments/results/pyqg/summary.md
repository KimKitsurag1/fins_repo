# Quasi-geostrophic flow, pyqg (exp61): counts over seeds

| row | n | own count = k | missing <= 0.05 | max kept quotient (median) | min broken quotient (median) | rotation plane (median) | scaling eigenvalues <= 1e-4 |
|---|---|---|---|---|---|---|---|
| exact q f | 10 | 9 | 10 | 4.5e-06 | - | 2.5e-07 | {2: 10} |
| exact q beta | 10 | 10 | 10 | 1.4e-05 | 9.0e-03 | 1.4e-02 | {1: 10} |
| exact q rd | 10 | 10 | 10 | 4.0e-07 | 3.4e-03 | 3.1e-10 | {1: 10} |
| exact q drag | 10 | 10 | 10 | 2.6e-07 | 5.6e-03 | 1.6e-08 | {1: 10} |
| exact psi f | 10 | 1 | 0 | - | - | 6.4e-05 | {1: 10} |
| exact psi rd | 10 | 10 | 10 | - | - | 8.5e-09 | {1: 10} |
| exact q f n64 | 10 | 0 | 0 | 6.5e-03 | - | 6.0e-07 | {1: 10} |
| fno q f | 3 | 0 | 0 | 1.7e-02 | - | 4.1e-03 | {0: 3} |
| fno q beta | 3 | 0 | 0 | 3.8e-03 | 1.1e-02 | 1.5e-02 | {0: 3} |
