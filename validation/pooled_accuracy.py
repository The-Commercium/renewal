"""How accurate is the 101 x 101 grid of pool_rates (spec section 7)?

Compares the Python implementation with an independent SciPy integration on a grid four times
finer in each direction over a box widened by 2 on every side. Writes pooled_accuracy.md.
Development only (needs numpy and scipy):  python validation/pooled_accuracy.py
"""
import os, sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, betaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python", "src"))
import renewal as rn
from renewal.rates import _linspace, _log_hyper

CASES = [
    ([3, 10, 1, 25], [40, 90, 15, 200]),
    ([2, 30, 5, 1, 12], [50, 60, 40, 30, 100]),
    ([0, 0, 1], [10, 25, 30]),
    ([12, 15, 9, 20, 14, 11, 30, 8], [100, 120, 80, 150, 110, 90, 260, 70]),
    ([45, 5], [60, 50]),
    ([0, 3, 0, 0], [8, 9, 15, 40]),
]


def reference(k, n, level=0.95):
    k, n = np.array(k, float), np.array(n, float)
    # the implementation's fine box, then widened by 2 on each side
    us, vs = _linspace(-10, 10, 41), _linspace(-30, 20, 101)
    coarse = [(u, v, _log_hyper(u, v, k.tolist(), n.tolist())) for u in us for v in vs]
    top = max(p[2] for p in coarse)
    keep = [p for p in coarse if p[2] > top - 20]
    u = np.linspace(max(min(p[0] for p in keep) - 2.5, -12), min(max(p[0] for p in keep) + 2.5, 12), 401)
    v = np.linspace(max(min(p[1] for p in keep) - 2.5, -40), min(max(p[1] for p in keep) + 2.5, 22), 401)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    mu, kap = 1 / (1 + np.exp(-U)), np.exp(Vv)
    a, b = mu * kap, (1 - mu) * kap
    lp = np.log(mu) + np.log1p(-mu) - Vv / 2
    for ki, ni in zip(k, n):
        lp += betaln(a + ki, b + ni - ki) - betaln(a, b)
    w = np.exp(lp - lp.max()).ravel()
    w /= w.sum()
    a, b = a.ravel(), b.ravel()
    sel = w > 1e-14
    w, a, b = w[sel] / w[sel].sum(), a[sel], b[sel]
    out = []
    for ki, ni in zip(k, n):
        A, B = a + ki, b + ni - ki
        mean = np.sum(w * A / (A + B))
        q = lambda p: brentq(lambda x: np.sum(w * betainc(A, B, x)) - p, 0.0, 1.0, xtol=1e-16, rtol=1e-15)
        out.append((mean, q((1 - level) / 2), q((1 + level) / 2)))
    return out


lines = ["# Accuracy of the pooled-rates grid", "",
         "Largest relative difference between `pool_rates` (101 x 101 grid) and an independent SciPy",
         "integration on a 401 x 401 grid over a box widened by 2 on every side (differences are",
         "taken relative to max(|reference|, 1e-4), since a lower bound can sit almost at zero).", "",
         "| groups | estimate | lower | upper |", "|---|---|---|---|"]
for k, n in CASES:
    got = rn.pool_rates(k, n).estimates
    ref = reference(k, n)
    diffs = [max(abs(getattr(g, f) - r[i]) / max(abs(r[i]), 1e-4) for g, r in zip(got, ref))
             for i, f in enumerate(("estimate", "lower", "upper"))]
    lines.append(f"| {len(k)} | {diffs[0]:.1e} | {diffs[1]:.1e} | {diffs[2]:.1e} |")
text = "\n".join(lines) + "\n"
open(os.path.join(HERE, "pooled_accuracy.md"), "w").write(text)
print(text)
