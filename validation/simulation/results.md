# Coverage of nominal 95% intervals

2000 simulated data sets per row. Subscribers join evenly over 12 periods and are
observed at the end (right-censored). Monte Carlo standard error of a coverage near 0.95 is
about 0.005. `boundary`: fits at the no-heterogeneity boundary
(alpha + beta > 1e5, spec 4.4); `failed`: no cancellation or renewal yet, or no convergence.
Boundary fits are scored for S, r and DEL but not for alpha and beta, which are
meaningless there; failed fits are not scored.

| true model | n | S(6) | S(12) | r(3) | DEL(0.01) | alpha | beta | boundary | failed |
|---|---|---|---|---|---|---|---|---|---|
| sBG, alpha 0.7, beta 1.2 | 50 | 0.958 | 0.968 | 0.950 | 0.901 | 0.978 | 0.972 | 3 | 0 |
| sBG, alpha 0.7, beta 1.2 | 200 | 0.948 | 0.946 | 0.943 | 0.928 | 0.951 | 0.953 | 0 | 0 |
| sBG, alpha 0.7, beta 1.2 | 1000 | 0.953 | 0.961 | 0.962 | 0.959 | 0.962 | 0.955 | 0 | 0 |
| sBG, alpha 0.67, beta 3.8 | 50 | 0.933 | 0.940 | 0.938 | 0.832 | 0.945 | 0.948 | 150 | 0 |
| sBG, alpha 0.67, beta 3.8 | 200 | 0.954 | 0.953 | 0.954 | 0.929 | 0.957 | 0.961 | 5 | 0 |
| sBG, alpha 0.67, beta 3.8 | 1000 | 0.938 | 0.948 | 0.944 | 0.945 | 0.954 | 0.961 | 0 | 0 |
| sBG, alpha 2, beta 6 | 50 | 0.934 | 0.889 | 0.931 | 0.852 | 0.933 | 0.942 | 277 | 0 |
| sBG, alpha 2, beta 6 | 200 | 0.950 | 0.955 | 0.950 | 0.917 | 0.953 | 0.956 | 25 | 0 |
| sBG, alpha 2, beta 6 | 1000 | 0.943 | 0.947 | 0.942 | 0.929 | 0.960 | 0.965 | 0 | 0 |
| BdW, alpha 1.4, beta 4, c 0.7 | 50 | 0.946 | 0.947 | 0.952 | 0.842 | 0.979 | 0.976 | 738 | 116 |
| BdW, alpha 1.4, beta 4, c 0.7 | 200 | 0.940 | 0.948 | 0.946 | 0.800 | 0.935 | 0.946 | 579 | 46 |
| BdW, alpha 1.4, beta 4, c 0.7 | 1000 | 0.949 | 0.943 | 0.949 | 0.836 | 0.916 | 0.914 | 222 | 1 |
