"""Reproduce Fader and Hardie (2007), "How to project customer retention".

Fit the sBG to years 0-7 of each segment, project years 8-12, compare with what happened.
Published estimates: regular alpha 0.704, beta 1.182; high end alpha 0.668, beta 3.806.
Writes results.md next to this file.   python validation/fader-hardie-2007/reproduce.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "src"))
import renewal as rn

DATA = {  # customers still active at the start of years 0..12, out of 1,000 (Table 1 of the paper)
    "regular": [1000, 631, 468, 382, 326, 289, 262, 241, 223, 207, 194, 183, 173],
    "high end": [1000, 869, 743, 653, 593, 551, 517, 491, 468, 445, 427, 409, 394],
}
PUBLISHED = {"regular": (0.704, 1.182), "high end": (0.668, 3.806)}

lines = ["# Fader and Hardie (2007), reproduced", "",
         "sBG fitted to years 0 to 7 of each segment; years 8 to 12 are a true hold-out.", "",
         "| segment | alpha (published) | alpha | beta (published) | beta | log-likelihood |",
         "|---|---|---|---|---|---|"]
fits = {}
for seg, alive in DATA.items():
    f = rn.fit_sbg(alive=alive[:8])
    fits[seg] = f
    pa, pb = PUBLISHED[seg]
    lines.append(f"| {seg} | {pa} | {f.alpha:.4f} | {pb} | {f.beta:.4f} | {f.loglik:.3f} |")
lines += ["", "Hold-out: customers still active, actual against projected (95% interval).", "",
          "| year | regular, actual | regular, projected | high end, actual | high end, projected |",
          "|---|---|---|---|---|"]
for t in range(8, 13):
    cells = []
    for seg in DATA:
        e = fits[seg].survival(t)
        cells += [str(DATA[seg][t]), f"{1000 * e.estimate:.0f} ({1000 * e.lower:.0f} to {1000 * e.upper:.0f})"]
    lines.append(f"| {t} | " + " | ".join(cells) + " |")
lines += ["", "Duration dependence (beta-discrete-Weibull, Fader et al. 2018) on the same years 0 to 7:", "",
          "| segment | c | 95% interval | LR test p-value (c = 1) |", "|---|---|---|---|"]
for seg, alive in DATA.items():
    b = rn.fit_bdw(alive=alive[:8])
    ci = b.params()["c"]
    lines.append(f"| {seg} | {b.c:.3f} | {ci.lower:.3f} to {ci.upper:.3f} | {rn.compare(fits[seg], b)['p_value']:.4f} |")
text = "\n".join(lines) + "\n"
open(os.path.join(HERE, "results.md"), "w").write(text)
print(text)
