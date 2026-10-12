"""Long-run value at the no-heterogeneity boundary: the BdW case with 1,000 subscribers.

Splits DEL (d = 0.01) by whether the fit landed on the boundary (alpha + beta > 1e5).
Writes boundary.md. Development only (needs numpy):  python validation/simulation/boundary.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "src"))
sys.path.insert(0, HERE)
import renewal as rn
from renewal.models import _residual
from study import simulate

REPS = 400
rng = np.random.default_rng(7)
truth = _residual("bdw", 1.4, 4.0, 0.7, 0.01, 0)
res = {True: [0, 0, []], False: [0, 0, []]}
for _ in range(REPS):
    p, a = simulate(rng, 1000, 1.4, 4.0, 0.7)
    f = rn.fit_bdw(p, a)
    if not f.converged and not f.at_boundary:
        continue
    e = f.discounted_lifetime(0.01)
    r = res[f.at_boundary]
    r[0] += 1
    r[1] += e.lower <= truth <= e.upper
    r[2].append(e.estimate / truth)
lines = ["# DEL at the no-heterogeneity boundary", "",
         f"True model BdW (alpha 1.4, beta 4, c 0.7), 1,000 subscribers joining over 12 periods, {REPS} data sets.",
         f"True DEL at d = 0.01: {truth:.2f}.", "",
         "| fits | count | median DEL / truth | coverage of the 95% interval |", "|---|---|---|---|"]
for b, label in ((True, "at the boundary"), (False, "interior")):
    n, h, ratio = res[b]
    lines.append(f"| {label} | {n} | {np.median(ratio):.3f} | {h / n:.3f} |")
text = "\n".join(lines) + "\n"
open(os.path.join(HERE, "boundary.md"), "w").write(text)
print(text)
