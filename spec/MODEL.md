# renewal: model specification

This file is the single source of truth for the three implementations (R, Python, TypeScript).
Each one must reproduce `conformance/vectors.json` within the tolerances stated there.

## 1. Setting and conventions

Time is counted in whole billing periods (weeks, months or years). A subscriber pays at the
start of every period in which they are subscribed. Period 1 is the first paid period.

- **Lifetime** `T ∈ {1, 2, …}` is the number of periods a subscriber pays for. `T = t` means the
  subscriber paid for periods `1..t` and did not renew for period `t + 1`.
- **Survival** `S(t) = P(T > t)` for `t = 0, 1, 2, …`, with `S(0) = 1`. `S(t)` is the share of a
  cohort that renews `t` times.
- **Churn probability** `P(T = t) = S(t − 1) − S(t)` for `t ≥ 1`.
- **Retention rate** `r(t) = S(t) / S(t − 1)` for `t ≥ 1`: of those who paid for period `t`, the
  share who renew for period `t + 1`.

## 2. Models

Each subscriber has a cancellation propensity `θ ∈ (0, 1)`. Across subscribers,
`θ ~ Beta(α, β)` with `α, β > 0`. `B(·,·)` is the beta function.

### 2.1 Shifted beta-geometric (sBG)

Fader and Hardie (2007). At every renewal a subscriber cancels with their own constant
probability `θ`.

    S(t)     = B(α, β + t) / B(α, β)
    P(T = t) = B(α + 1, β + t − 1) / B(α, β)
    r(t)     = (β + t − 1) / (α + β + t − 1)

`r(t)` rises with `t` even though no individual's `θ` changes: the subscribers most likely to
cancel leave first (sorting, not loyalty).

### 2.2 Beta-discrete-Weibull (BdW)

Fader, Hardie, Liu, Davin and Steenburgh (2018). An individual survives `t` renewals with
probability `(1 − θ)^(t^c)`, `c > 0`. `c < 1` means an individual's own cancellation probability
falls with tenure, `c > 1` that it rises. `c = 1` is the sBG.

    S(t)     = B(α, β + t^c) / B(α, β)
    P(T = t) = S(t − 1) · (1 − exp(log S(t) − log S(t − 1)))      (computed with expm1)

## 3. Data and likelihood

Two input forms, reduced to the same internal counts.

**Individual records.** For each subscriber: `periods` (integer ≥ 1, periods paid so far) and
`active` (true if the subscriber has not cancelled: their next renewal has not happened yet).

- cancelled, `periods = t`: contributes `log P(T = t)`
- active, `periods = t`: contributes `log S(t − 1)` (they have renewed `t − 1` times)

**Cohort survivor counts.** `alive = (A_0, A_1, …, A_k)`: `A_0` subscribers started, `A_t` of them
renewed `t` times. This equals `A_{t−1} − A_t` cancelled records with `periods = t` for
`t = 1..k`, plus `A_k` active records with `periods = k + 1`.

The log-likelihood is the sum of the contributions. The model is not identifiable without at
least one cancellation and at least one observed renewal; implementations raise an error.

## 4. Estimation

Maximum likelihood on `x = (log α, log β)` for the sBG and `x = (log α, log β, log c)` for the
BdW, minimising `f(x) = −log L`. `f(x) = +∞` when any `|x_i| > 20` or `f` is not finite: beyond
`e^20` the log-gamma differences lose their meaning, and a fit that drifts there is at the
boundary of section 4.4 anyway.

1. **Nelder–Mead** from `x = 0` (all parameters 1). Initial simplex: `x` and `x + 0.5·e_i`.
   Coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5. Outside contraction
   when the reflected point beats the worst, inside contraction otherwise. Stop when the spread
   of `f` over the simplex is below `1e−10` and every vertex is within `1e−8` (max norm) of the
   best, or after 5000 iterations.
2. **Newton polish.** Gradient by central differences with `h = 1e−5`, Hessian by central
   differences with `h = 1e−3`. Step `−H⁻¹g`, halved up to 30 times until `f` decreases. Stop
   when the step is below `1e−10` (max norm) or after 20 steps. Skipped if `H` is not positive
   definite.
3. **Converged** means `H` is positive definite at the end and `max|g| < 1e−3`.
4. **Boundary.** When `α + β > 1e5` the fit has reached the no-heterogeneity boundary: the data
   cannot tell the subscribers' cancellation propensities apart, and the model is close to a
   geometric (sBG) or discrete-Weibull (BdW) model with one `θ = α/(α + β)` for everyone. The
   likelihood is flat along that ridge, so the fit is reported as at the boundary and usually
   not converged. Survival and retention within the observed periods remain usable. `α`, `β` and
   long-run quantities (DEL, DERL) do not: a fit that cannot see heterogeneity understates the
   loyal tail, so its long-run value is biased low (see `validation/simulation/results.md`).

The covariance of `x` is `H⁻¹` (observed information), with `H` from Richardson extrapolation
of the central-difference Hessians at `h = 1e−3` and `h = 2e−3`: `H = (4·H(1e−3) − H(2e−3)) / 3`.
With a plain central difference, round-off and truncation leave relative errors near `1e−5` in
poorly identified models; the extrapolated Hessian is good to about `1e−6`.

## 5. Derived quantities and intervals

For a quantity `g(x)` the interval uses the delta method on a transformed scale:
`logit` for probabilities (`S`, `P`, `r`), `log` for positive quantities (`α`, `β`, `c`, DEL, DERL,
expected lifetime). With `u = transform(g)` and `∇u` by central differences (`h = 1e−5`):

    se(u) = sqrt(∇uᵀ · V · ∇u),   interval = inverse_transform(u ± z · se(u))

where `V = H⁻¹` and `z` is the standard normal quantile at `(1 + level) / 2`.

### 5.1 Discounted expected lifetime (DEL)

Expected number of payments of a new subscriber, discounted at rate `d ≥ 0` per period, the first
payment undiscounted. With `ρ = 1 / (1 + d)`:

    DEL(d) = Σ_{t ≥ 0} S(t) ρ^t

For the sBG this equals `₂F₁(1, β; α + β; ρ)`. At `d = 0` it is the expected lifetime
`E[T] = (α + β − 1) / (α − 1)`, finite only if `α > 1`; implementations return infinity otherwise.
For the BdW, `d > 0` is required.

### 5.2 Discounted expected residual lifetime (DERL)

Expected discounted future payments of a subscriber who has paid for `n ≥ 1` periods and is at the
renewal point for period `n + 1` (that payment, if made, undiscounted):

    DERL(d, n) = Σ_{t ≥ 0} S(n + t) / S(n − 1) · ρ^t

For the sBG this equals `(β + n − 1)/(α + β + n − 1) · ₂F₁(1, β + n; α + β + n; ρ)` (Fader and
Hardie 2010), and at `d = 0` it is `(β + n − 1)/(α − 1)` for `α > 1`.

Series stop when a term falls below `1e−15` of the running sum, with an error after `10⁶` terms.

Multiply DEL or DERL by the net revenue per period to get a value in money.

## 6. Comparing models

`compare(sbg, bdw)`: likelihood-ratio statistic `LR = 2(ℓ_BdW − ℓ_sBG)` with one degree of freedom,
`p = P(χ²₁ > LR) = erfc(sqrt(LR / 2))`. AIC = `2k − 2ℓ`, BIC = `k log N − 2ℓ`, `N` = subscribers.

## 7. Conversion rates

**One rate.** `k` successes in `n` trials (for example trials that became paid). The interval is
the Jeffreys interval: the equal-tailed quantiles of `Beta(k + ½, n − k + ½)`, with the lower bound
set to 0 when `k = 0` and the upper bound to 1 when `k = n` (Brown, Cai and DasGupta 2001).

**Pooled rates.** Groups `i = 1..m` with `k_i` successes in `n_i` trials (products, countries,
paywalls). Hierarchical model: `p_i ~ Beta(a, b)`, `k_i ~ Binomial(n_i, p_i)`, with the hyperprior
`p(a, b) ∝ (a + b)^(−5/2)` of Gelman et al., *Bayesian Data Analysis* (3rd ed., section 5.3). The
hyperparameters are integrated out, not plugged in, so the intervals carry the uncertainty about
how alike the groups are. (Plug-in empirical Bayes gives intervals of near zero width when the
groups differ no more than chance would make them.)

With `μ = a/(a + b)`, `κ = a + b`, `u = logit μ`, `v = log κ`, the log hyperposterior is

    L(u, v) = log μ + log(1 − μ) − v/2 + Σ_i [ log B(a + k_i, b + n_i − k_i) − log B(a, b) ]

1. **Coarse grid**: `u` from −10 to 10 (41 points), `v` from −30 to 20 (101 points). Keep the points
   with `L > max L − 20`; their bounding box, widened by 0.5 on each side and clipped to the coarse
   range, is the fine box.
2. **Fine grid**: 101 × 101 evenly spaced points over the fine box, weights `exp(L − max L)`.
3. **Prune**: sort by weight, keep the heaviest points until they hold `1 − 1e−10` of the total,
   renormalise.
4. Each group's posterior is the mixture `Σ_j w_j Beta(a_j + k_i, b_j + n_i − k_i)`. The estimate is
   its mean; the interval bounds are its quantiles at `(1 ∓ level)/2`, found by Newton steps inside
   a bisection bracket to `1e−13`.
5. Also reported: the posterior mean of `μ` and the posterior median of `κ` (the concentration:
   large means the groups' true rates are close).

**Properness.** As `κ → 0` each group with `0 < k_i < n_i` contributes a factor of order `κ` to the
likelihood, and the hyperprior grows like `κ^(−1/2)` in `v`. The posterior is proper only if at
least one group has both successes and failures; implementations raise an error otherwise. The
lower limit `v = −30` keeps the truncated tail below about `1e−6` of the mass in the worst proper
case (one such group, tail `∝ e^(v/2)`); `validation/pooled_accuracy.md` measures the grid error.

## References

- Fader, P. S. and Hardie, B. G. S. (2007). How to project customer retention. *Journal of
  Interactive Marketing* 21(1), 76–90.
- Fader, P. S. and Hardie, B. G. S. (2010). Customer-base valuation in a contractual setting: the
  perils of ignoring heterogeneity. *Marketing Science* 29(1), 85–93.
- Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J. and Steenburgh, T. (2018). "How to project
  customer retention" revisited: the role of duration dependence. *Journal of Interactive
  Marketing* 43, 1–16.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013).
  *Bayesian Data Analysis*, 3rd ed. Chapman & Hall/CRC.
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion.
  *Statistical Science* 16(2), 101–133.
