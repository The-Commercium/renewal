"""Contractual retention models: shifted beta-geometric (sBG) and beta-discrete-Weibull (BdW)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ._numeric import chisq1_sf, gradient, lbeta, minimise, norm_ppf

_MAX_TERMS = 1_000_000


@dataclass(frozen=True)
class Estimate:
    """A point estimate with the bounds of its interval."""

    estimate: float
    lower: float
    upper: float


# ---- distribution functions (spec section 2) ---------------------------------------------


def _log_survival(t: float, alpha: float, beta: float, c: float) -> float:
    if t <= 0:
        return 0.0
    return lbeta(alpha, beta + t**c) - lbeta(alpha, beta)


def _log_churn(t: int, alpha: float, beta: float, c: float, model: str) -> float:
    if model == "sbg":
        return lbeta(alpha + 1.0, beta + t - 1.0) - lbeta(alpha, beta)
    prev = _log_survival(t - 1, alpha, beta, c)
    return prev + math.log(-math.expm1(_log_survival(t, alpha, beta, c) - prev))


# ---- data (spec section 3) ---------------------------------------------------------------


def _counts(periods, active, alive) -> tuple[dict[int, list[int]], int]:
    """Reduce either input form to {t: [cancelled with periods t, active with periods t]}."""
    counts: dict[int, list[int]] = {}
    if alive is not None:
        if periods is not None or active is not None:
            raise ValueError("give either periods and active, or alive, not both")
        a = [int(v) for v in alive]
        if len(a) < 2 or a[0] <= 0 or any(v < 0 for v in a):
            raise ValueError("alive needs at least two non-negative counts and alive[0] > 0")
        if any(a[i] > a[i - 1] for i in range(1, len(a))):
            raise ValueError("alive must not increase")
        for t in range(1, len(a)):
            if a[t - 1] - a[t]:
                counts.setdefault(t, [0, 0])[0] += a[t - 1] - a[t]
        if a[-1]:
            counts.setdefault(len(a), [0, 0])[1] += a[-1]
        n = a[0]
    else:
        if periods is None or active is None:
            raise ValueError("give periods and active, or alive")
        p, act = list(periods), list(active)
        if len(p) != len(act) or not p:
            raise ValueError("periods and active must have the same, non-zero length")
        for t, on in zip(p, act):
            if int(t) != t or t < 1:
                raise ValueError("periods must be whole numbers of at least 1")
            counts.setdefault(int(t), [0, 0])[1 if on else 0] += 1
        n = len(p)
    churned = sum(v[0] for v in counts.values())
    renewed = sum(v[0] + v[1] for t, v in counts.items() if t >= 2)
    if churned == 0:
        raise ValueError("no cancellations observed: the model cannot be estimated")
    if renewed == 0:
        raise ValueError("no renewals observed: the model cannot be estimated")
    return counts, n


def _loglik(alpha: float, beta: float, c: float, model: str, counts) -> float:
    ll = 0.0
    for t, (dead, cens) in counts.items():
        if dead:
            ll += dead * _log_churn(t, alpha, beta, c, model)
        if cens:
            ll += cens * _log_survival(t - 1, alpha, beta, c)
    return ll


# ---- fitting (spec section 4) ------------------------------------------------------------


def fit_sbg(periods: Sequence[int] | None = None, active: Sequence[bool] | None = None, *,
            alive: Sequence[int] | None = None) -> "RetentionFit":
    """Fit the shifted beta-geometric model by maximum likelihood.

    Give individual records (``periods`` paid so far and whether each subscriber is still
    ``active``) or cohort survivor counts (``alive``: the starting cohort, then how many
    renewed once, twice, ...).
    """
    return _fit("sbg", periods, active, alive)


def fit_bdw(periods: Sequence[int] | None = None, active: Sequence[bool] | None = None, *,
            alive: Sequence[int] | None = None) -> "RetentionFit":
    """Fit the beta-discrete-Weibull model (the sBG with duration dependence ``c``)."""
    return _fit("bdw", periods, active, alive)


def _fit(model, periods, active, alive) -> "RetentionFit":
    counts, n = _counts(periods, active, alive)
    k = 2 if model == "sbg" else 3

    def nll(x):
        c = math.exp(x[2]) if k == 3 else 1.0
        return -_loglik(math.exp(x[0]), math.exp(x[1]), c, model, counts)

    res = minimise(nll, [0.0] * k)
    return RetentionFit(model, res["x"], -res["value"], res["cov"], n, res["converged"],
                        res["iterations"], counts)


class RetentionFit:
    """A fitted sBG or BdW model. Every quantity comes with a delta-method interval."""

    def __init__(self, model, x, loglik, cov, n, converged, iterations, counts):
        self.model = model
        self._x = list(x)
        self.loglik = loglik
        self.vcov = cov
        self.n = n
        self.converged = converged
        self.iterations = iterations
        self._counts = counts
        self.k = len(x)
        self.aic = 2 * self.k - 2 * loglik
        self.bic = self.k * math.log(n) - 2 * loglik

    # parameters
    @property
    def alpha(self) -> float:
        return math.exp(self._x[0])

    @property
    def beta(self) -> float:
        return math.exp(self._x[1])

    @property
    def c(self) -> float:
        return math.exp(self._x[2]) if self.k == 3 else 1.0

    @property
    def at_boundary(self) -> bool:
        """True when alpha + beta > 1e5: no detectable heterogeneity between subscribers."""
        return self.alpha + self.beta > 1e5

    def _par(self, x):
        return math.exp(x[0]), math.exp(x[1]), (math.exp(x[2]) if self.k == 3 else 1.0)

    def _interval(self, fn, transform: str, level: float) -> Estimate:
        try:
            value = fn(self._x)
        except (ValueError, OverflowError, ZeroDivisionError):
            return Estimate(math.nan, math.nan, math.nan)
        if not math.isfinite(value) or self.vcov is None:
            return Estimate(value, math.nan, math.nan)
        if transform == "logit":
            fwd, back = _logit, _expit
        else:
            fwd, back = math.log, _exp
        try:
            g = gradient(lambda x: fwd(fn(x)), self._x)
        except (ValueError, OverflowError, ZeroDivisionError):
            return Estimate(value, math.nan, math.nan)
        if not all(math.isfinite(v) for v in g):
            return Estimate(value, math.nan, math.nan)
        var = sum(g[i] * self.vcov[i][j] * g[j] for i in range(self.k) for j in range(self.k))
        z = norm_ppf((1.0 + level) / 2.0)
        u, se = fwd(value), math.sqrt(max(var, 0.0))
        return Estimate(value, back(u - z * se), back(u + z * se))

    def params(self, level: float = 0.95) -> dict[str, Estimate]:
        names = ["alpha", "beta", "c"][: self.k]
        return {name: self._interval(lambda x, i=i: math.exp(x[i]), "log", level)
                for i, name in enumerate(names)}

    # distribution
    def survival(self, t: int, level: float = 0.95) -> Estimate:
        """Share of a new cohort that renews ``t`` times, S(t)."""
        t = _check_t(t, 0)
        if t == 0:
            return Estimate(1.0, 1.0, 1.0)
        return self._interval(lambda x: math.exp(_log_survival(t, *self._par(x))), "logit", level)

    def churn_probability(self, t: int, level: float = 0.95) -> Estimate:
        """Probability that a new subscriber pays for exactly ``t`` periods, P(T = t)."""
        t = _check_t(t, 1)
        return self._interval(lambda x: math.exp(_log_churn(t, *self._par(x), self.model)),
                              "logit", level)

    def retention(self, t: int, level: float = 0.95) -> Estimate:
        """Of those who paid for period ``t``, the share who renew for period ``t + 1``."""
        t = _check_t(t, 1)

        def fn(x):
            a, b, c = self._par(x)
            return math.exp(_log_survival(t, a, b, c) - _log_survival(t - 1, a, b, c))

        return self._interval(fn, "logit", level)

    def expected_cohort(self, start: int, periods: int) -> list[float]:
        """Expected number of a cohort of ``start`` still subscribed after 0..periods renewals."""
        a, b, c = self._par(self._x)
        return [start * math.exp(_log_survival(t, a, b, c)) for t in range(periods + 1)]

    # value
    def _check_d(self, d: float) -> None:
        if d < 0:
            raise ValueError("the discount rate d must be >= 0")
        if d == 0 and self.model != "sbg":
            raise ValueError("the BdW needs a discount rate d > 0")

    def discounted_lifetime(self, d: float, level: float = 0.95) -> Estimate:
        """DEL: expected discounted payments of a new subscriber (first one undiscounted)."""
        self._check_d(d)
        return self._interval(lambda x: _residual(self.model, *self._par(x), d, 0), "log", level)

    def residual_lifetime(self, d: float, n: int, level: float = 0.95) -> Estimate:
        """DERL: expected discounted future payments of a subscriber who has paid ``n`` periods."""
        n = _check_t(n, 1)
        self._check_d(d)
        return self._interval(lambda x: _residual(self.model, *self._par(x), d, n), "log", level)

    def expected_lifetime(self, level: float = 0.95) -> Estimate:
        """E[T], the undiscounted expected number of payments (sBG only; infinite if alpha <= 1)."""
        if self.model != "sbg":
            raise ValueError("the BdW has no closed-form expected lifetime; use discounted_lifetime(d > 0)")
        return self.discounted_lifetime(0.0, level)

    def median_lifetime(self) -> int:
        """The smallest t with S(t) <= 0.5."""
        a, b, c = self._par(self._x)
        for t in range(1, _MAX_TERMS):
            if _log_survival(t, a, b, c) <= math.log(0.5):
                return t
        raise ArithmeticError("median lifetime beyond 10^6 periods")

    def summary(self, level: float = 0.95) -> str:
        name = {"sbg": "shifted beta-geometric (sBG)", "bdw": "beta-discrete-Weibull (BdW)"}[self.model]
        lines = [f"{name}, {self.n} subscribers", f"log-likelihood {self.loglik:.4f}, "
                 f"AIC {self.aic:.2f}, BIC {self.bic:.2f}, converged {self.converged}"]
        for key, e in self.params(level).items():
            lines.append(f"  {key:<5} {e.estimate:10.4f}   [{e.lower:.4f}, {e.upper:.4f}]")
        if self.at_boundary:
            lines.append("  at the no-heterogeneity boundary (alpha + beta > 1e5): the data cannot tell"
                         " subscribers apart yet. Survival within the data is usable; alpha, beta and"
                         " long-run value (DEL, DERL) are not: missing heterogeneity biases them low.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _logit(v: float) -> float:
    return math.log(v) - math.log1p(-v)


def _expit(u: float) -> float:
    if u >= 0:
        return 1.0 / (1.0 + math.exp(-u))
    e = math.exp(u)
    return e / (1.0 + e)


def _exp(u: float) -> float:
    return math.exp(u) if u < 709.0 else math.inf


def _residual(model: str, a: float, b: float, c: float, d: float, n: int) -> float:
    """DEL when n == 0, DERL(d, n) otherwise (spec 5.1, 5.2)."""
    if d < 0:
        raise ValueError("the discount rate d must be >= 0")
    if d == 0:
        if model != "sbg":
            raise ValueError("the BdW needs a discount rate d > 0")
        if a <= 1.0:
            return math.inf
        return (a + b - 1.0) / (a - 1.0) if n == 0 else (b + n - 1.0) / (a - 1.0)
    log_rho = -math.log1p(d)
    base = _log_survival(n - 1, a, b, c) if n >= 1 else 0.0
    total = 0.0
    for t in range(_MAX_TERMS):
        term = math.exp(_log_survival(n + t, a, b, c) - base + t * log_rho)
        total += term
        if term < 1e-15 * total:
            return total
    raise ArithmeticError("discounted lifetime did not converge within 10^6 terms")


def _check_t(t, low: int) -> int:
    if int(t) != t or t < low:
        raise ValueError(f"t must be a whole number of at least {low}")
    return int(t)


def compare(sbg: RetentionFit, bdw: RetentionFit) -> dict:
    """Likelihood-ratio test of the sBG (c = 1) against the BdW, with AIC and BIC."""
    if sbg.model != "sbg" or bdw.model != "bdw":
        raise ValueError("compare(sbg_fit, bdw_fit) expects an sBG fit then a BdW fit")
    if sbg.n != bdw.n or sbg._counts != bdw._counts:
        raise ValueError("both fits must use the same data")
    lr = max(2.0 * (bdw.loglik - sbg.loglik), 0.0)
    return {"lr": lr, "df": 1, "p_value": chisq1_sf(lr),
            "aic": {"sbg": sbg.aic, "bdw": bdw.aic}, "bic": {"sbg": sbg.bic, "bdw": bdw.bic}}


def survival_curve(t: int | Iterable[int], alpha: float, beta: float, c: float = 1.0):
    """S(t) for given parameters (c = 1 is the sBG)."""
    if isinstance(t, Iterable):
        return [survival_curve(v, alpha, beta, c) for v in t]
    return math.exp(_log_survival(_check_t(t, 0), alpha, beta, c))
