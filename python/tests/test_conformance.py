import json
import math
import os

import pytest

import renewal as rn
from renewal.models import _residual

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "..", "conformance", "vectors.json")) as fh:
    V = json.load(fh)
TOL = V["tolerances"]


def close(got, want, rel):
    if math.isinf(want):
        return math.isinf(got)
    return abs(got - want) <= rel * max(abs(want), 1e-300)


@pytest.mark.parametrize("case", V["survival"])
def test_survival(case):
    got = rn.survival_curve(case["t"], case["alpha"], case["beta"], case["c"])
    assert close(got, case["value"], TOL["closed_form_rel"])


@pytest.mark.parametrize("case", V["residual"])
def test_residual(case):
    got = _residual(case["model"], case["alpha"], case["beta"], case["c"], case["d"], case["n"])
    assert close(got, case["value"], TOL["closed_form_rel"])


@pytest.mark.parametrize("case", V["rate_interval"])
def test_rate_interval(case):
    e = rn.rate_interval(case["k"], case["n"], case["level"])
    assert e.estimate == case["estimate"]
    assert abs(e.lower - case["lower"]) <= TOL["jeffreys_abs"]
    assert abs(e.upper - case["upper"]) <= TOL["jeffreys_abs"]


def _fit(case):
    fn = rn.fit_sbg if case["model"] == "sbg" else rn.fit_bdw
    d = case["data"]
    return fn(alive=d["alive"]) if "alive" in d else fn(d["periods"], d["active"])


@pytest.mark.parametrize("case", V["fits"], ids=[c["name"] for c in V["fits"]])
def test_fit(case):
    f = _fit(case)
    e = case["expected"]
    assert f.converged
    for key in ("alpha", "beta", "c"):
        assert close(getattr(f, key), e[key], TOL["fit_param_rel"]), key
    assert abs(f.loglik - e["loglik"]) <= TOL["fit_loglik_abs"]
    q, r = case["intervals"], TOL["interval_rel"]

    def check(est, want):
        assert all(close(g, w, r) for g, w in zip((est.estimate, est.lower, est.upper), want)), (est, want)

    for name, want in q["params"].items():
        check(f.params()[name], want)
    for t, want in q["survival"].items():
        check(f.survival(int(t)), want)
    for t, want in q["churn_probability"].items():
        check(f.churn_probability(int(t)), want)
    for t, want in q["retention"].items():
        check(f.retention(int(t)), want)
    for d, want in q["discounted_lifetime"].items():
        check(f.discounted_lifetime(float(d)), want)
    for key, want in q["residual_lifetime"].items():
        d, n = key.split("|")
        check(f.residual_lifetime(float(d), int(n)), want)


@pytest.mark.parametrize("case", V["pool_rates"])
def test_pool_rates(case):
    r = rn.pool_rates(case["k"], case["n"], case["level"])
    assert close(r.mean, case["mean"], TOL["pooled_rel"])
    assert close(r.concentration, case["concentration"], TOL["pooled_rel"])
    for e, want in zip(r.estimates, case["estimates"]):
        assert all(close(g, w, TOL["pooled_rel"]) for g, w in zip((e.estimate, e.lower, e.upper), want))
