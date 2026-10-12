"""Every output of the Python implementation on the conformance inputs, as JSON on stdout."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "src"))
import renewal as rn
from renewal.models import _residual

V = json.load(open(os.path.join(HERE, "..", "vectors.json")))
out = {}
for i, c in enumerate(V["survival"]):
    out[f"survival/{i}"] = rn.survival_curve(c["t"], c["alpha"], c["beta"], c["c"])
for i, c in enumerate(V["residual"]):
    out[f"residual/{i}"] = _residual(c["model"], c["alpha"], c["beta"], c["c"], c["d"], c["n"])
for i, c in enumerate(V["rate_interval"]):
    e = rn.rate_interval(c["k"], c["n"])
    out[f"jeffreys/{i}"] = [e.lower, e.upper]
for c in V["fits"]:
    fn = rn.fit_sbg if c["model"] == "sbg" else rn.fit_bdw
    d = c["data"]
    f = fn(alive=d["alive"]) if "alive" in d else fn(d["periods"], d["active"])
    k = c["name"]
    out[f"{k}/par"] = [f.alpha, f.beta, f.c, f.loglik]
    for nm, e in f.params().items():
        out[f"{k}/param/{nm}"] = [e.estimate, e.lower, e.upper]
    for t in (1, 3, 12):
        e = f.survival(t); out[f"{k}/S/{t}"] = [e.estimate, e.lower, e.upper]
    for t in (1, 6):
        e = f.retention(t); out[f"{k}/r/{t}"] = [e.estimate, e.lower, e.upper]
    for d_ in (0.01, 0.1):
        e = f.discounted_lifetime(d_); out[f"{k}/DEL/{d_}"] = [e.estimate, e.lower, e.upper]
    e = f.residual_lifetime(0.1, 5); out[f"{k}/DERL"] = [e.estimate, e.lower, e.upper]
for i, c in enumerate(V["pool_rates"]):
    r = rn.pool_rates(c["k"], c["n"])
    out[f"pooled/{i}"] = [r.mean, r.concentration] + [v for e in r.estimates for v in (e.estimate, e.lower, e.upper)]
print(json.dumps(out))
