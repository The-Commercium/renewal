import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { fitBdw, fitSbg, poolRates, rateInterval, residualValue, survivalCurve, type Estimate } from "../src/index.ts";

const V = JSON.parse(readFileSync(new URL("../../conformance/vectors.json", import.meta.url), "utf8"));
const TOL = V.tolerances;

const close = (got: number, want: number, rel: number) =>
  want === null || !Number.isFinite(want) ? !Number.isFinite(got) : Math.abs(got - want) <= rel * Math.max(Math.abs(want), 1e-300);

test("survival matches the 30-digit references", () => {
  for (const c of V.survival) assert.ok(close(survivalCurve(c.t, c.alpha, c.beta, c.c), c.value, TOL.closed_form_rel), JSON.stringify(c));
});

test("DEL and DERL match the 30-digit references", () => {
  for (const c of V.residual) {
    assert.ok(close(residualValue(c.model, c.alpha, c.beta, c.c, c.d, c.n), c.value, TOL.closed_form_rel), JSON.stringify(c));
  }
});

test("Jeffreys intervals match", () => {
  for (const c of V.rate_interval) {
    const e = rateInterval(c.k, c.n, c.level);
    assert.equal(e.estimate, c.estimate);
    assert.ok(Math.abs(e.lower - c.lower) <= TOL.jeffreys_abs, `${e.lower} ${c.lower}`);
    assert.ok(Math.abs(e.upper - c.upper) <= TOL.jeffreys_abs, `${e.upper} ${c.upper}`);
  }
});

for (const c of V.fits) {
  test(`fit ${c.name} and every interval match the references`, () => {
    const fn = c.model === "sbg" ? fitSbg : fitBdw;
    const f = fn(c.data.alive ? { alive: c.data.alive } : { periods: c.data.periods, active: c.data.active });
    assert.ok(f.converged);
    for (const key of ["alpha", "beta", "c"] as const) assert.ok(close(f[key], c.expected[key], TOL.fit_param_rel), key);
    assert.ok(Math.abs(f.loglik - c.expected.loglik) <= TOL.fit_loglik_abs);
    const check = (e: Estimate, want: number[], what: string) => {
      const got = [e.estimate, e.lower, e.upper];
      assert.ok(got.every((g, i) => close(g, want[i], TOL.interval_rel)), `${what}: ${got} vs ${want}`);
    };
    const q = c.intervals;
    const pars = f.params();
    for (const [nm, want] of Object.entries(q.params)) check(pars[nm], want as number[], nm);
    for (const [t, want] of Object.entries(q.survival)) check(f.survival(Number(t)), want as number[], `S(${t})`);
    for (const [t, want] of Object.entries(q.churn_probability)) check(f.churnProbability(Number(t)), want as number[], `P(${t})`);
    for (const [t, want] of Object.entries(q.retention)) check(f.retention(Number(t)), want as number[], `r(${t})`);
    for (const [d, want] of Object.entries(q.discounted_lifetime)) check(f.discountedLifetime(Number(d)), want as number[], `DEL(${d})`);
    for (const [key, want] of Object.entries(q.residual_lifetime)) {
      const [d, n] = key.split("|").map(Number);
      check(f.residualLifetime(d, n), want as number[], `DERL(${key})`);
    }
  });
}

test("pooled rates match the reference implementation", () => {
  for (const c of V.pool_rates) {
    const r = poolRates(c.k, c.n, c.level);
    assert.ok(close(r.mean, c.mean, TOL.pooled_rel), `mean ${r.mean} ${c.mean}`);
    assert.ok(close(r.concentration, c.concentration, TOL.pooled_rel), `kappa ${r.concentration} ${c.concentration}`);
    r.estimates.forEach((e, i) => {
      const got = [e.estimate, e.lower, e.upper];
      assert.ok(got.every((g, j) => close(g, c.estimates[i][j], TOL.pooled_rel)), `${got} vs ${c.estimates[i]}`);
    });
  }
});
