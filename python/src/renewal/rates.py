"""Conversion rates with honest intervals for small counts (spec section 7)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ._numeric import beta_cdf, beta_ppf, lbeta
from .models import Estimate


def _check(k: int, n: int) -> None:
    if int(k) != k or int(n) != n or n < 1 or k < 0 or k > n:
        raise ValueError("need whole numbers with 0 <= k <= n and n >= 1")


def rate_interval(k: int, n: int, level: float = 0.95) -> Estimate:
    """k successes in n trials: the rate k/n with its Jeffreys interval."""
    _check(k, n)
    tail = (1.0 - level) / 2.0
    lower = 0.0 if k == 0 else beta_ppf(tail, k + 0.5, n - k + 0.5)
    upper = 1.0 if k == n else beta_ppf(1.0 - tail, k + 0.5, n - k + 0.5)
    return Estimate(k / n, lower, upper)


@dataclass(frozen=True)
class PooledRates:
    """Hierarchical rates for several groups that share a Beta(a, b) population."""

    mean: float
    """Posterior mean of the population rate a / (a + b)."""
    concentration: float
    """Posterior median of a + b: how tightly the groups' true rates cluster."""
    raw: list[float]
    estimates: list[Estimate]


def _linspace(lo: float, hi: float, num: int) -> list[float]:
    return [lo + (hi - lo) * i / (num - 1) for i in range(num)]


def _log_hyper(u: float, v: float, k, n) -> float:
    mu = 1.0 / (1.0 + math.exp(-u))
    kappa = math.exp(v)
    a, b = mu * kappa, (1.0 - mu) * kappa
    out = math.log(mu) + math.log1p(-mu) - v / 2.0
    lab = lbeta(a, b)
    for ki, ni in zip(k, n):
        out += lbeta(a + ki, b + ni - ki) - lab
    return out


def _grid(k, n) -> list[tuple[float, float, float]]:
    """Hyperposterior on (logit mean, log concentration) as (a, b, weight) triples (spec 7)."""
    us, vs = _linspace(-10.0, 10.0, 41), _linspace(-30.0, 20.0, 101)
    coarse = [(u, v, _log_hyper(u, v, k, n)) for u in us for v in vs]
    top = max(p[2] for p in coarse)
    keep = [p for p in coarse if p[2] > top - 20.0]
    u_lo = max(min(p[0] for p in keep) - 0.5, -10.0)
    u_hi = min(max(p[0] for p in keep) + 0.5, 10.0)
    v_lo = max(min(p[1] for p in keep) - 0.5, -30.0)
    v_hi = min(max(p[1] for p in keep) + 0.5, 20.0)
    fine = [(u, v, _log_hyper(u, v, k, n))
            for u in _linspace(u_lo, u_hi, 101) for v in _linspace(v_lo, v_hi, 101)]
    top = max(p[2] for p in fine)
    pts = sorted(((u, v, math.exp(lp - top)) for u, v, lp in fine), key=lambda p: -p[2])
    total = sum(p[2] for p in pts)
    out, acc = [], 0.0
    for u, v, w in pts:
        out.append((u, v, w))
        acc += w
        if acc >= (1.0 - 1e-10) * total:
            break
    return [(1.0 / (1.0 + math.exp(-u)) * math.exp(v), (1.0 - 1.0 / (1.0 + math.exp(-u))) * math.exp(v),
             w / acc, v) for u, v, w in out]


def _mixture_quantile(q: float, comps) -> float:
    """Quantile of sum_j w_j Beta(a_j, b_j), safeguarded Newton inside a bisection bracket."""
    lo, hi = 0.0, 1.0
    x = sum(w * a / (a + b) for a, b, w in comps)
    for _ in range(200):
        cdf = sum(w * beta_cdf(x, a, b) for a, b, w in comps) - q
        if abs(cdf) < 1e-14:
            return x
        if cdf < 0:
            lo = x
        else:
            hi = x
        pdf = sum(w * math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - lbeta(a, b))
                  for a, b, w in comps) if 0.0 < x < 1.0 else 0.0
        nxt = x - cdf / pdf if pdf > 0 else math.nan
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - x) < 1e-13:
            return nxt
        x = nxt
    return x


def pool_rates(k: Sequence[int], n: Sequence[int], level: float = 0.95) -> PooledRates:
    """Rates of several groups, each pulled towards the others as far as the data justify.

    Full Bayes for the beta-binomial hierarchy with the hyperprior p(a, b) ~ (a + b)^(-5/2)
    (Gelman et al., Bayesian Data Analysis, section 5.3), integrated on a grid, so the
    intervals carry the uncertainty about how alike the groups are.
    """
    k, n = list(k), list(n)
    if len(k) != len(n) or len(k) < 2:
        raise ValueError("k and n need the same length, at least two groups")
    for ki, ni in zip(k, n):
        _check(ki, ni)
    if not any(0 < ki < ni for ki, ni in zip(k, n)):
        raise ValueError("at least one group needs both successes and failures (otherwise the posterior is improper)")
    grid = _grid(k, n)
    mean = sum(w * a / (a + b) for a, b, w, _ in grid)
    acc, median_v = 0.0, grid[0][3]
    for a, b, w, v in sorted(grid, key=lambda p: p[3]):
        acc += w
        if acc >= 0.5:
            median_v = v
            break
    tail = (1.0 - level) / 2.0
    est = []
    for ki, ni in zip(k, n):
        comps = [(a + ki, b + ni - ki, w) for a, b, w, _ in grid]
        m = sum(w * a / (a + b) for a, b, w in comps)
        est.append(Estimate(m, _mixture_quantile(tail, comps), _mixture_quantile(1.0 - tail, comps)))
    return PooledRates(mean, math.exp(median_v), [ki / ni for ki, ni in zip(k, n)], est)
