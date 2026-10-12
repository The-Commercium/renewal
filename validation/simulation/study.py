"""How often do the 95% intervals contain the truth? A coverage study.

Subscribers join evenly over 12 periods and are observed at the end, so most records are
right-censored, as in a young product. For each true model and sample size we simulate 2,000
data sets, fit with the Python implementation, and count how often each 95% interval contains
the true value. Results go to results.csv and results.md next to this file.

Development only (needs numpy):  python validation/simulation/study.py
"""

import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "src"))

import renewal as rn  # noqa: E402
from renewal.models import _residual  # noqa: E402

REPS = int(os.environ.get("REPS", 2000))
JOIN_PERIODS = 12
SIZES = (50, 200, 1000)
TRUTHS = [
    ("sBG, alpha 0.7, beta 1.2", "sbg", 0.7, 1.2, 1.0),
    ("sBG, alpha 0.67, beta 3.8", "sbg", 0.67, 3.8, 1.0),
    ("sBG, alpha 2, beta 6", "sbg", 2.0, 6.0, 1.0),
    ("BdW, alpha 1.4, beta 4, c 0.7", "bdw", 1.4, 4.0, 0.7),
]
D = 0.01  # discount rate per period for DEL


def simulate(rng, n, alpha, beta, c):
    joined = (np.arange(n) * JOIN_PERIODS) // n
    window = JOIN_PERIODS - joined
    theta = rng.beta(alpha, beta, size=n)
    periods = np.empty(n, dtype=int)
    active = np.empty(n, dtype=bool)
    for i in range(n):
        t = 1
        while True:
            if t >= window[i]:
                periods[i], active[i] = window[i], True
                break
            if rng.random() > (1 - theta[i]) ** (t**c - (t - 1) ** c):
                periods[i], active[i] = t, False
                break
            t += 1
    return periods.tolist(), active.tolist()


def main():
    rng = np.random.default_rng(20261012)
    rows = []
    for label, model, a, b, c in TRUTHS:
        truth = {
            "S(6)": rn.survival_curve(6, a, b, c),
            "S(12)": rn.survival_curve(12, a, b, c),
            "r(3)": rn.survival_curve(3, a, b, c) / rn.survival_curve(2, a, b, c),
            "DEL(0.01)": _residual(model, a, b, c, D, 0),
            "alpha": a,
            "beta": b,
        }
        for n in SIZES:
            t0 = time.time()
            hits = {k: 0 for k in truth}
            done = {k: 0 for k in truth}
            rel_err = {k: [] for k in truth}
            failed = 0
            boundary = 0
            for _ in range(REPS):
                periods, active = simulate(rng, n, a, b, c)
                try:
                    f = (rn.fit_sbg if model == "sbg" else rn.fit_bdw)(periods, active)
                except ValueError:
                    failed += 1
                    continue
                est = {"S(6)": f.survival(6), "S(12)": f.survival(12), "r(3)": f.retention(3),
                       "DEL(0.01)": f.discounted_lifetime(D)}
                if f.at_boundary:
                    boundary += 1  # alpha and beta are meaningless here; the rest is still scored
                elif not f.converged:
                    failed += 1
                    continue
                else:
                    p = f.params()
                    est.update(alpha=p["alpha"], beta=p["beta"])
                for key, e in est.items():
                    if math.isnan(e.lower) or math.isnan(e.upper):
                        continue
                    done[key] += 1
                    hits[key] += e.lower <= truth[key] <= e.upper
                    rel_err[key].append(abs(e.estimate - truth[key]) / truth[key])
            for key in truth:
                rows.append({
                    "truth": label, "n": n, "quantity": key,
                    "coverage": hits[key] / done[key] if done[key] else float("nan"),
                    "median_rel_error": float(np.median(rel_err[key])) if rel_err[key] else float("nan"),
                    "fits": done[key], "failed": failed, "boundary": boundary,
                })
            print(f"{label:32s} n={n:5d} failed={failed:4d} boundary={boundary:4d} "
                  + " ".join(f"{k}={hits[k] / max(done[k], 1):.3f}" for k in truth)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["S(6)", "S(12)", "r(3)", "DEL(0.01)", "alpha", "beta"]
    lines = ["# Coverage of nominal 95% intervals", "",
             f"{REPS} simulated data sets per row. Subscribers join evenly over {JOIN_PERIODS} periods and are",
             "observed at the end (right-censored). Monte Carlo standard error of a coverage near 0.95 is",
             f"about {math.sqrt(0.95 * 0.05 / REPS):.3f}. `boundary`: fits at the no-heterogeneity boundary",
             "(alpha + beta > 1e5, spec 4.4); `failed`: no cancellation or renewal yet, or no convergence.",
             "Boundary fits are scored for S, r and DEL but not for alpha and beta, which are",
             "meaningless there; failed fits are not scored.", "",
             "| true model | n | " + " | ".join(keys) + " | boundary | failed |",
             "|---|---|" + "---|" * len(keys) + "---|---|"]
    for label, *_ in TRUTHS:
        for n in SIZES:
            sel = {r["quantity"]: r for r in rows if r["truth"] == label and r["n"] == n}
            lines.append(f"| {label} | {n} | " + " | ".join(f"{sel[k]['coverage']:.3f}" for k in keys)
                         + f" | {sel[keys[0]]['boundary']} | {sel[keys[0]]['failed']} |")
    with open(os.path.join(HERE, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
