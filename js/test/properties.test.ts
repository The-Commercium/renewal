import assert from "node:assert/strict";
import { test } from "node:test";

import { compare, fitBdw, fitSbg, poolRates, rateInterval } from "../src/index.ts";

const regular = [1000, 631, 468, 382, 326, 289, 262, 241];

test("the published Fader and Hardie (2007) estimates come back", () => {
  const f = fitSbg({ alive: regular });
  assert.equal(f.alpha.toFixed(3), "0.704");
  assert.equal(f.beta.toFixed(3), "1.182");
});

test("cohort counts and individual records give the same fit", () => {
  const periods: number[] = [];
  const active: boolean[] = [];
  for (let t = 1; t < regular.length; t++) for (let i = 0; i < regular[t - 1] - regular[t]; i++) { periods.push(t); active.push(false); }
  for (let i = 0; i < regular[regular.length - 1]; i++) { periods.push(regular.length); active.push(true); }
  assert.ok(Math.abs(fitSbg({ alive: regular }).loglik - fitSbg({ periods, active }).loglik) < 1e-9);
});

test("the BdW nests the sBG", () => {
  const s = fitSbg({ alive: regular });
  const b = fitBdw({ alive: regular });
  assert.ok(b.loglik >= s.loglik - 1e-9);
  const p = compare(s, b).pValue;
  assert.ok(p >= 0 && p <= 1);
});

test("retention rises under the sBG; intervals contain estimates", () => {
  const f = fitSbg({ alive: regular });
  for (let t = 1; t < 14; t++) assert.ok(f.retention(t).estimate < f.retention(t + 1).estimate);
  const e = f.discountedLifetime(0.1);
  assert.ok(e.lower < e.estimate && e.estimate < e.upper);
  assert.equal(f.expectedLifetime().estimate, Infinity);
});

test("bad input is refused", () => {
  assert.throws(() => fitSbg({ alive: [100] }));
  assert.throws(() => fitSbg({ alive: [100, 120] }));
  assert.throws(() => fitSbg({ alive: [100, 100, 100] }));
  assert.throws(() => fitSbg({ periods: [1, 1, 1], active: [false, false, false] }));
  assert.throws(() => rateInterval(5, 3));
  assert.throws(() => fitBdw({ alive: regular }).discountedLifetime(0));
});

test("pooling moves an extreme small group towards the others", () => {
  const r = poolRates([1, 9, 30], [20, 30, 100]);
  assert.ok(r.estimates[0].estimate > r.raw[0]);
  for (const e of r.estimates) assert.ok(e.lower < e.estimate && e.estimate < e.upper);
});
