"""Special functions and the optimiser shared by every model (spec sections 4, 5 and 7)."""

from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def norm_ppf(p: float) -> float:
    return _NORMAL.inv_cdf(p)


def chisq1_sf(x: float) -> float:
    """P(chi-squared with one degree of freedom > x)."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            return h
    raise ArithmeticError("incomplete beta: continued fraction did not converge")


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) by bisection to full double precision."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- small dense linear algebra ----------------------------------------------------------


def _cholesky(m: list[list[float]]) -> list[list[float]] | None:
    n = len(m)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = m[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if s <= 0.0 or not math.isfinite(s):
                    return None
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def inverse_pd(m: list[list[float]]) -> list[list[float]] | None:
    """Inverse of a symmetric positive-definite matrix, or None if it is not one."""
    low = _cholesky(m)
    if low is None:
        return None
    n = len(m)
    inv = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [1.0 if i == col else 0.0 for i in range(n)]
        y = [0.0] * n
        for i in range(n):
            y[i] = (e[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
        for i in range(n):
            inv[i][col] = x[i]
    return inv


def gradient(f, x: list[float], h: float = 1e-5) -> list[float]:
    g = []
    for i in range(len(x)):
        up, dn = list(x), list(x)
        up[i] += h
        dn[i] -= h
        g.append((f(up) - f(dn)) / (2.0 * h))
    return g


def hessian(f, x: list[float], h: float = 1e-3) -> list[list[float]]:
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                up, dn = list(x), list(x)
                up[i] += h
                dn[i] -= h
                v = (f(up) - 2.0 * f0 + f(dn)) / (h * h)
            else:
                pp, pm, mp, mm = list(x), list(x), list(x), list(x)
                pp[i] += h; pp[j] += h
                pm[i] += h; pm[j] -= h
                mp[i] -= h; mp[j] += h
                mm[i] -= h; mm[j] -= h
                v = (f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h * h)
            out[i][j] = out[j][i] = v
    return out


# ---- optimiser (spec section 4) ----------------------------------------------------------


def _nelder_mead(f, x0: list[float]) -> tuple[list[float], float, int]:
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (0.5 if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    it = 0
    while it < 5000:
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        spread = vals[-1] - vals[0]
        size = max(max(abs(p[j] - pts[0][j]) for j in range(n)) for p in pts[1:])
        if spread < 1e-10 and size < 1e-8:
            break
        it += 1
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        worst = pts[-1]
        xr = [cen[j] + (cen[j] - worst[j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2.0 * (cen[j] - worst[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
            continue
        if fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
            continue
        if fr < vals[-1]:
            xc = [cen[j] + 0.5 * (xr[j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc <= fr:
                pts[-1], vals[-1] = xc, fc
                continue
        else:
            xc = [cen[j] + 0.5 * (worst[j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
                continue
        best = pts[0]
        for i in range(1, n + 1):
            pts[i] = [best[j] + 0.5 * (pts[i][j] - best[j]) for j in range(n)]
            vals[i] = f(pts[i])
    i = min(range(n + 1), key=lambda k: vals[k])
    return pts[i], vals[i], it


def _safe(f):
    def g(x):
        if max(abs(v) for v in x) > 20.0:
            return math.inf
        try:
            v = f(x)
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.inf
        return v if math.isfinite(v) else math.inf
    return g


def minimise(f, x0: list[float]) -> dict:
    """Nelder-Mead, then Newton polish. Returns x, value, covariance (inverse Hessian), converged."""
    f = _safe(f)
    x, fx, iters = _nelder_mead(f, x0)
    for _ in range(20):
        h = hessian(f, x)
        inv = inverse_pd(h)
        if inv is None:
            break
        g = gradient(f, x)
        step = [-sum(inv[i][j] * g[j] for j in range(len(x))) for i in range(len(x))]
        if max(abs(s) for s in step) < 1e-10:
            break
        scale, moved = 1.0, False
        for _ in range(30):
            cand = [x[i] + scale * step[i] for i in range(len(x))]
            fc = f(cand)
            if fc < fx:
                x, fx, moved = cand, fc, True
                break
            scale *= 0.5
        if not moved:
            break
    h1, h2 = hessian(f, x, 1e-3), hessian(f, x, 2e-3)
    h = [[(4.0 * a - b) / 3.0 for a, b in zip(r1, r2)] for r1, r2 in zip(h1, h2)]
    cov = inverse_pd(h)
    g = gradient(f, x)
    converged = cov is not None and max(abs(v) for v in g) < 1e-3
    return {"x": x, "value": fx, "cov": cov, "converged": converged, "iterations": iters}
