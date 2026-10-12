import math

import pytest

import renewal as rn

FH_REGULAR = [1000, 631, 468, 382, 326, 289, 262, 241, 223, 207, 194, 183, 173]


def test_published_estimates():
    """Fader and Hardie (2007): alpha 0.704, beta 1.182 (regular); 0.668, 3.806 (high end)."""
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    assert round(f.alpha, 3) == 0.704 and round(f.beta, 3) == 1.182
    g = rn.fit_sbg(alive=[1000, 869, 743, 653, 593, 551, 517, 491])
    assert round(g.alpha, 3) == 0.668 and round(g.beta, 3) == 3.806


def test_cohort_and_records_agree():
    alive = FH_REGULAR[:8]
    periods, active = [], []
    for t in range(1, len(alive)):
        periods += [t] * (alive[t - 1] - alive[t])
        active += [False] * (alive[t - 1] - alive[t])
    periods += [len(alive)] * alive[-1]
    active += [True] * alive[-1]
    a, b = rn.fit_sbg(alive=alive), rn.fit_sbg(periods, active)
    assert abs(a.loglik - b.loglik) < 1e-9 and abs(a.alpha - b.alpha) < 1e-9


def test_bdw_contains_sbg():
    assert rn.survival_curve(7, 0.8, 2.0, 1.0) == pytest.approx(rn.survival_curve(7, 0.8, 2.0), rel=1e-15)
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    b = rn.fit_bdw(alive=FH_REGULAR[:8])
    assert b.loglik >= f.loglik - 1e-9
    cmp = rn.compare(f, b)
    assert 0.0 <= cmp["p_value"] <= 1.0


def test_retention_rises_under_sbg():
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    r = [f.retention(t).estimate for t in range(1, 15)]
    assert all(x < y for x, y in zip(r, r[1:]))


def test_intervals_contain_estimates():
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    for e in (f.survival(5), f.retention(2), f.discounted_lifetime(0.1), f.residual_lifetime(0.1, 3)):
        assert e.lower < e.estimate < e.upper


def test_infinite_mean_lifetime():
    f = rn.fit_sbg(alive=FH_REGULAR[:8])  # alpha < 1
    assert math.isinf(f.expected_lifetime().estimate)


def test_expected_cohort():
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    cohort = f.expected_cohort(1000, 12)
    assert cohort[0] == 1000 and abs(cohort[12] - FH_REGULAR[12]) < 10


def test_median_lifetime():
    f = rn.fit_sbg(alive=FH_REGULAR[:8])
    m = f.median_lifetime()
    assert f.survival(m).estimate <= 0.5 < f.survival(m - 1).estimate


@pytest.mark.parametrize("bad", [
    dict(alive=[100]), dict(alive=[100, 120]), dict(alive=[100, 100, 100]),
    dict(periods=[1, 1, 1], active=[False, False, False]), dict(periods=[0, 2], active=[True, False]),
])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        rn.fit_sbg(**bad)


def test_bdw_needs_discounting():
    b = rn.fit_bdw(alive=FH_REGULAR[:8])
    with pytest.raises(ValueError):
        b.discounted_lifetime(0.0)


def test_rate_interval_edges():
    assert rn.rate_interval(0, 10).lower == 0.0
    assert rn.rate_interval(10, 10).upper == 1.0
    with pytest.raises(ValueError):
        rn.rate_interval(5, 3)


def test_pooling_shrinks():
    r = rn.pool_rates([1, 9, 30], [20, 30, 100])
    raw = r.raw
    for e, x in zip(r.estimates, raw):
        assert min(raw) <= e.estimate <= max(raw)
        assert e.lower < e.estimate < e.upper
    # the extreme group moves towards the others
    assert r.estimates[0].estimate > raw[0]


def test_pooling_refuses_improper_posterior():
    with pytest.raises(ValueError):
        rn.pool_rates([0, 5], [10, 5])
