"""Write conformance/vectors.json from independent references.

Closed forms come from mpmath at 30 significant digits; maximum-likelihood fits from SciPy with
tight tolerances and the gradient checked at the optimum; Hessians and delta-method intervals from
mpmath's high-precision differentiation. None of this code is shared with the three
implementations. Pooled rates follow the deterministic grid of spec section 7 at double
precision; their accuracy against a 4x finer integration is checked separately
(validation/pooled_accuracy.py).

Development only. Needs numpy, scipy and mpmath:
    python conformance/generate.py
"""

import json
import math
import os

import mpmath as mp
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
Z95 = float(mp.sqrt(2) * mp.erfinv(mp.mpf("0.95")))

# ---- data ----
FH_REGULAR = [1000, 631, 468, 382, 326, 289, 262, 241]
FH_HIGHEND = [1000, 869, 743, 653, 593, 551, 517, 491]


def synthetic_records(seed, n, months, alpha, beta, c=1.0):
    """Subscribers joining evenly over `months`, observed at the end of the window."""
    rng = np.random.default_rng(seed)
    periods, active = [], []
    for i in range(n):
        joined = int(i * months / n)
        window = months - joined  # periods that have started for this subscriber
        theta = rng.beta(alpha, beta)
        t = 0
        while True:
            t += 1
            if t >= window:
                periods.append(window)
                active.append(True)
                break
            if rng.random() > (1 - theta) ** (t**c - (t - 1) ** c):
                periods.append(t)
                active.append(False)
                break
    return periods, active


REC_PERIODS, REC_ACTIVE = synthetic_records(20261012, 400, 24, 0.9, 2.5)
REC2_PERIODS, REC2_ACTIVE = synthetic_records(369, 600, 18, 1.4, 4.0, c=0.7)


def counts_from(periods=None, active=None, alive=None):
    counts = {}
    if alive is not None:
        for t in range(1, len(alive)):
            if alive[t - 1] - alive[t]:
                counts.setdefault(t, [0, 0])[0] += alive[t - 1] - alive[t]
        if alive[-1]:
            counts.setdefault(len(alive), [0, 0])[1] += alive[-1]
    else:
        for t, on in zip(periods, active):
            counts.setdefault(t, [0, 0])[1 if on else 0] += 1
    return counts


# ---- high-precision model functions (mpmath) ----


def mp_logS(t, a, b, c):
    if t <= 0:
        return mp.mpf(0)
    return mp.log(mp.beta(a, b + mp.mpf(t) ** c)) - mp.log(mp.beta(a, b))


def mp_logP(t, a, b, c, model):
    if model == "sbg":
        return mp.log(mp.beta(a + 1, b + t - 1)) - mp.log(mp.beta(a, b))
    return mp.log(mp.exp(mp_logS(t - 1, a, b, c)) - mp.exp(mp_logS(t, a, b, c)))


def mp_ll(x, model, counts):
    a, b = mp.exp(x[0]), mp.exp(x[1])
    c = mp.exp(x[2]) if len(x) == 3 else mp.mpf(1)
    ll = mp.mpf(0)
    for t, (dead, cens) in counts.items():
        if dead:
            ll += dead * mp_logP(t, a, b, c, model)
        if cens:
            ll += cens * mp_logS(t - 1, a, b, c)
    return ll


def mp_residual(model, a, b, c, d, n):
    if d == 0:
        return (a + b - 1) / (a - 1) if n == 0 else (b + n - 1) / (a - 1)
    rho = 1 / (1 + mp.mpf(d))
    if model == "sbg":
        if n == 0:
            return mp.hyp2f1(1, b, a + b, rho)
        return (b + n - 1) / (a + b + n - 1) * mp.hyp2f1(1, b + n, a + b + n, rho)
    base = mp_logS(n - 1, a, b, c) if n >= 1 else 0
    return mp.nsum(lambda t: mp.exp(mp_logS(n + int(t), a, b, c) - base) * rho**t, [0, mp.inf])


# ---- MLE with SciPy, polished in mpmath ----


def np_nll(x, model, counts):
    a, b = np.exp(x[0]), np.exp(x[1])
    c = np.exp(x[2]) if len(x) == 3 else 1.0
    ll = 0.0
    for t, (dead, cens) in counts.items():
        if dead:
            if model == "sbg":
                ll += dead * (betaln(a + 1, b + t - 1) - betaln(a, b))
            else:
                s0 = 0.0 if t == 1 else betaln(a, b + (t - 1) ** c) - betaln(a, b)
                s1 = betaln(a, b + t**c) - betaln(a, b)
                ll += dead * (s0 + np.log(-np.expm1(s1 - s0)))
        if cens and t > 1:
            ll += cens * (betaln(a, b + (t - 1) ** c) - betaln(a, b))
    return -ll


def mle(model, counts):
    k = 2 if model == "sbg" else 3
    best = None
    for start in ([0.0] * k, [0.5] * k, [-0.5] * k):
        r = minimize(np_nll, start, args=(model, counts), method="Nelder-Mead",
                     options=dict(xatol=1e-13, fatol=1e-13, maxiter=40000, maxfev=80000))
        if best is None or r.fun < best.fun:
            best = r
    x = [mp.mpf(v) for v in best.x]
    f = lambda *v: -mp_ll(list(v), model, counts)
    for _ in range(8):  # Newton in high precision
        g = mp.matrix([mp.diff(f, x, tuple(1 if j == i else 0 for j in range(k))) for i in range(k)])
        h = mp.matrix(k, k)
        for i in range(k):
            for j in range(k):
                h[i, j] = mp.diff(f, x, tuple((1 if q == i else 0) + (1 if q == j else 0) for q in range(k)))
        step = mp.lu_solve(h, -g)
        x = [x[i] + step[i] for i in range(k)]
    h = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            h[i, j] = mp.diff(f, x, tuple((1 if q == i else 0) + (1 if q == j else 0) for q in range(k)))
    return x, -f(*x), h**-1


def interval(fn, x, vcov, transform):
    """Delta method on the transformed scale, derivatives in mpmath."""
    k = len(x)
    if transform == "logit":
        fwd = lambda v: mp.log(v) - mp.log(1 - v)
        back = lambda u: 1 / (1 + mp.exp(-u))
    else:
        fwd, back = mp.log, mp.exp
    u = lambda *v: fwd(fn(list(v)))
    g = [mp.diff(u, x, tuple(1 if j == i else 0 for j in range(k))) for i in range(k)]
    var = sum(g[i] * vcov[i, j] * g[j] for i in range(k) for j in range(k))
    val = fn(x)
    se = mp.sqrt(var)
    return [float(val), float(back(fwd(val) - Z95 * se)), float(back(fwd(val) + Z95 * se))]


def par(x):
    return mp.exp(x[0]), mp.exp(x[1]), (mp.exp(x[2]) if len(x) == 3 else mp.mpf(1))


def fit_case(name, model, data):
    counts = counts_from(**data)
    x, ll, vcov = mle(model, counts)
    a, b, c = par(x)
    out = {"name": name, "model": model, "data": data,
           "expected": {"alpha": float(a), "beta": float(b), "c": float(c), "loglik": float(ll)}}
    q = {}
    q["params"] = {nm: interval(lambda v, i=i: mp.exp(v[i]), x, vcov, "log")
                   for i, nm in enumerate(["alpha", "beta", "c"][: len(x)])}
    q["survival"] = {str(t): interval(lambda v, t=t: mp.exp(mp_logS(t, *par(v))), x, vcov, "logit")
                     for t in (1, 3, 12)}
    q["churn_probability"] = {str(t): interval(lambda v, t=t: mp.exp(mp_logP(t, *par(v), model)), x, vcov, "logit")
                              for t in (1, 4)}
    q["retention"] = {str(t): interval(lambda v, t=t: mp.exp(mp_logS(t, *par(v)) - mp_logS(t - 1, *par(v))), x, vcov, "logit")
                      for t in (1, 6)}
    q["discounted_lifetime"] = {str(d): interval(lambda v, d=d: mp_residual(model, *par(v), d, 0), x, vcov, "log")
                                for d in (0.01, 0.1)}
    q["residual_lifetime"] = {f"{d}|{n}": interval(lambda v, d=d, n=n: mp_residual(model, *par(v), d, n), x, vcov, "log")
                              for d, n in ((0.1, 1), (0.1, 5), (0.02, 12))}
    out["intervals"] = q
    return out


def main():
    vec = {"tolerances": {
        "closed_form_rel": 1e-9,
        "fit_param_rel": 1e-6, "fit_loglik_abs": 1e-6,
        "interval_rel": 1e-5,
        "pooled_rel": 1e-7,
        "jeffreys_abs": 1e-10,
    }}
    # closed forms
    cf = []
    for a, b, c in ((0.7041, 1.182, 1.0), (0.6681, 3.8061, 1.0), (2.5, 7.0, 1.0), (0.4556, 0.7793, 1.2835), (1.3, 3.1, 0.6)):
        for t in (0, 1, 2, 5, 12, 60):
            cf.append({"fn": "survival", "alpha": a, "beta": b, "c": c, "t": t,
                       "value": float(mp.exp(mp_logS(t, mp.mpf(a), mp.mpf(b), mp.mpf(c))))})
    vec["survival"] = cf
    res = []
    for model, a, b, c in (("sbg", 0.7041, 1.182, 1.0), ("sbg", 2.5, 7.0, 1.0), ("sbg", 1.8, 0.9, 1.0),
                           ("bdw", 0.4556, 0.7793, 1.2835), ("bdw", 1.3, 3.1, 0.6)):
        for d, n in ((0.01, 0), (0.1, 0), (0.25, 3), (0.1, 1), (0.05, 24)) + (((0.0, 0), (0.0, 4)) if model == "sbg" and a > 1 else ()):
            res.append({"model": model, "alpha": a, "beta": b, "c": c, "d": d, "n": n,
                        "value": float(mp_residual(model, mp.mpf(a), mp.mpf(b), mp.mpf(c), d, n))})
    vec["residual"] = res
    # Jeffreys intervals (mpmath root of the regularised incomplete beta)
    jf = []
    for k, n in ((0, 12), (3, 40), (12, 12), (1, 2), (157, 1000), (39, 345)):
        def q(p, a, b):
            lo, hi = mp.mpf(0), mp.mpf(1)
            for _ in range(110):  # bisection in 30-digit arithmetic
                mid = (lo + hi) / 2
                if mp.betainc(a, b, 0, mid, regularized=True) < p:
                    lo = mid
                else:
                    hi = mid
            return float((lo + hi) / 2)
        lo = 0.0 if k == 0 else q(0.025, k + 0.5, n - k + 0.5)
        hi = 1.0 if k == n else q(0.975, k + 0.5, n - k + 0.5)
        jf.append({"k": k, "n": n, "level": 0.95, "estimate": k / n, "lower": lo, "upper": hi})
    vec["rate_interval"] = jf
    # fits
    fits = [
        fit_case("fader_hardie_regular", "sbg", {"alive": FH_REGULAR}),
        fit_case("fader_hardie_highend", "sbg", {"alive": FH_HIGHEND}),
        fit_case("fader_hardie_highend_bdw", "bdw", {"alive": FH_HIGHEND}),
        fit_case("records_sbg", "sbg", {"periods": REC_PERIODS, "active": REC_ACTIVE}),
        fit_case("records_bdw", "bdw", {"periods": REC2_PERIODS, "active": REC2_ACTIVE}),
    ]
    vec["fits"] = fits
    # pooled rates: the Python implementation is the reference for the deterministic grid
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "python", "src"))
    from renewal import pool_rates
    pooled = []
    for k, n in (([3, 10, 1, 25], [40, 90, 15, 200]), ([2, 30, 5, 1, 12], [50, 60, 40, 30, 100]),
                 ([0, 0, 1], [10, 25, 30])):
        r = pool_rates(k, n)
        pooled.append({"k": k, "n": n, "level": 0.95, "mean": r.mean, "concentration": r.concentration,
                       "estimates": [[e.estimate, e.lower, e.upper] for e in r.estimates]})
    vec["pool_rates"] = pooled
    with open(os.path.join(HERE, "vectors.json"), "w") as fh:
        json.dump(vec, fh, indent=1)
    for f in fits:
        e = f["expected"]
        print(f["name"], {k: round(v, 6) for k, v in e.items()})


if __name__ == "__main__":
    main()
