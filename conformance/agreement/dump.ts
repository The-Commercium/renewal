// Every output of the TypeScript implementation on the conformance inputs, as JSON on stdout.
import { readFileSync } from "node:fs";
import { fitBdw, fitSbg, poolRates, rateInterval, residualValue, survivalCurve, type Estimate } from "../../js/src/index.ts";

const V = JSON.parse(readFileSync(new URL("../vectors.json", import.meta.url), "utf8"));
const out: Record<string, number | number[]> = {};
const e3 = (e: Estimate) => [e.estimate, e.lower, e.upper];
V.survival.forEach((c: any, i: number) => { out[`survival/${i}`] = survivalCurve(c.t, c.alpha, c.beta, c.c); });
V.residual.forEach((c: any, i: number) => { out[`residual/${i}`] = residualValue(c.model, c.alpha, c.beta, c.c, c.d, c.n); });
V.rate_interval.forEach((c: any, i: number) => { const e = rateInterval(c.k, c.n); out[`jeffreys/${i}`] = [e.lower, e.upper]; });
for (const c of V.fits) {
  const fn = c.model === "sbg" ? fitSbg : fitBdw;
  const f = fn(c.data.alive ? { alive: c.data.alive } : { periods: c.data.periods, active: c.data.active });
  const k = c.name;
  out[`${k}/par`] = [f.alpha, f.beta, f.c, f.loglik];
  for (const [nm, e] of Object.entries(f.params())) out[`${k}/param/${nm}`] = e3(e);
  for (const t of [1, 3, 12]) out[`${k}/S/${t}`] = e3(f.survival(t));
  for (const t of [1, 6]) out[`${k}/r/${t}`] = e3(f.retention(t));
  for (const d of [0.01, 0.1]) out[`${k}/DEL/${d}`] = e3(f.discountedLifetime(d));
  out[`${k}/DERL`] = e3(f.residualLifetime(0.1, 5));
}
V.pool_rates.forEach((c: any, i: number) => {
  const r = poolRates(c.k, c.n);
  out[`pooled/${i}`] = [r.mean, r.concentration, ...r.estimates.flatMap(e3)];
});
console.log(JSON.stringify(out));
