"""Largest relative disagreement between any two implementations, per group of outputs.

    python conformance/agreement/compare.py
"""
import json, math, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
runs = {
    "python": [sys.executable, os.path.join(HERE, "dump.py")],
    "r": ["Rscript", os.path.join(HERE, "dump.R")],
    "typescript": ["node", os.path.join(HERE, "dump.ts")],
}
outs = {k: json.loads(subprocess.run(v, capture_output=True, text=True, check=True).stdout) for k, v in runs.items()}
groups = {}
for key in outs["python"]:
    group = key.split("/")[0] if "/" in key and not key.startswith(("fader", "records")) else key.split("/")[1] if key.count("/") else key
    vals = {lang: outs[lang][key] for lang in outs}
    vecs = {lang: (v if isinstance(v, list) else [v]) for lang, v in vals.items()}
    worst = 0.0
    langs = list(vecs)
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            for a, b in zip(vecs[langs[i]], vecs[langs[j]]):
                if a is None or b is None or (isinstance(a, float) and math.isinf(a)):
                    continue
                worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1e-300))
    groups[group] = max(groups.get(group, 0.0), worst)
for g, w in sorted(groups.items()):
    print(f"{g:12s} {w:.1e}")
print(f"{'overall':12s} {max(groups.values()):.1e}")
