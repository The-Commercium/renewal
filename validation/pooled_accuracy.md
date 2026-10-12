# Accuracy of the pooled-rates grid

Largest relative difference between `pool_rates` (101 x 101 grid) and an independent SciPy
integration on a 401 x 401 grid over a box widened by 2 on every side (differences are
taken relative to max(|reference|, 1e-4), since a lower bound can sit almost at zero).

| groups | estimate | lower | upper |
|---|---|---|---|
| 4 | 2.0e-05 | 1.4e-03 | 2.2e-04 |
| 5 | 6.0e-10 | 7.5e-10 | 1.8e-09 |
| 3 | 3.0e-05 | 4.5e-06 | 6.1e-04 |
| 8 | 1.0e-05 | 1.4e-04 | 3.9e-04 |
| 2 | 8.7e-11 | 7.5e-11 | 2.2e-10 |
| 4 | 1.2e-06 | 2.6e-06 | 1.3e-05 |
