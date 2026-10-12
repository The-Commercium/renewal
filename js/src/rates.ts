// Conversion rates with honest intervals for small counts (spec section 7).

import type { Estimate } from "./models.ts";
import { betaCdf, betaPpf, lbeta } from "./numeric.ts";

function check(k: number, n: number): void {
  if (!Number.isInteger(k) || !Number.isInteger(n) || n < 1 || k < 0 || k > n) {
    throw new RangeError("need whole numbers with 0 <= k <= n and n >= 1");
  }
}

/** k successes in n trials: the rate k/n with its Jeffreys interval. */
export function rateInterval(k: number, n: number, level = 0.95): Estimate {
  check(k, n);
  const tail = (1 - level) / 2;
  return {
    estimate: k / n,
    lower: k === 0 ? 0 : betaPpf(tail, k + 0.5, n - k + 0.5),
    upper: k === n ? 1 : betaPpf(1 - tail, k + 0.5, n - k + 0.5),
  };
}

export interface PooledRates {
  /** Posterior mean of the population rate a / (a + b). */
  mean: number;
  /** Posterior median of a + b: how tightly the groups' true rates cluster. */
  concentration: number;
  raw: number[];
  estimates: Estimate[];
}

const linspace = (lo: number, hi: number, num: number) =>
  Array.from({ length: num }, (_, i) => lo + ((hi - lo) * i) / (num - 1));

function logHyper(u: number, v: number, k: number[], n: number[]): number {
  const mu = 1 / (1 + Math.exp(-u));
  const kappa = Math.exp(v);
  const a = mu * kappa;
  const b = (1 - mu) * kappa;
  let out = Math.log(mu) + Math.log1p(-mu) - v / 2;
  const lab = lbeta(a, b);
  for (let i = 0; i < k.length; i++) out += lbeta(a + k[i], b + n[i] - k[i]) - lab;
  return out;
}

interface Point { a: number; b: number; w: number; v: number }

function grid(k: number[], n: number[]): Point[] {
  const coarse: [number, number, number][] = [];
  for (const u of linspace(-10, 10, 41)) for (const v of linspace(-30, 20, 101)) coarse.push([u, v, logHyper(u, v, k, n)]);
  let top = Math.max(...coarse.map((p) => p[2]));
  const keep = coarse.filter((p) => p[2] > top - 20);
  const uLo = Math.max(Math.min(...keep.map((p) => p[0])) - 0.5, -10);
  const uHi = Math.min(Math.max(...keep.map((p) => p[0])) + 0.5, 10);
  const vLo = Math.max(Math.min(...keep.map((p) => p[1])) - 0.5, -30);
  const vHi = Math.min(Math.max(...keep.map((p) => p[1])) + 0.5, 20);
  const fine: [number, number, number][] = [];
  for (const u of linspace(uLo, uHi, 101)) for (const v of linspace(vLo, vHi, 101)) fine.push([u, v, logHyper(u, v, k, n)]);
  top = Math.max(...fine.map((p) => p[2]));
  const pts = fine.map(([u, v, lp]) => [u, v, Math.exp(lp - top)]).sort((p, q) => q[2] - p[2]);
  const total = pts.reduce((s, p) => s + p[2], 0);
  const out: number[][] = [];
  let acc = 0;
  for (const p of pts) {
    out.push(p);
    acc += p[2];
    if (acc >= (1 - 1e-10) * total) break;
  }
  return out.map(([u, v, w]) => {
    const mu = 1 / (1 + Math.exp(-u));
    return { a: mu * Math.exp(v), b: (1 - mu) * Math.exp(v), w: w / acc, v };
  });
}

function mixtureQuantile(q: number, comps: { a: number; b: number; w: number }[]): number {
  let lo = 0;
  let hi = 1;
  let x = comps.reduce((s, c) => s + (c.w * c.a) / (c.a + c.b), 0);
  for (let i = 0; i < 200; i++) {
    const cdf = comps.reduce((s, c) => s + c.w * betaCdf(x, c.a, c.b), 0) - q;
    if (Math.abs(cdf) < 1e-14) return x;
    if (cdf < 0) lo = x;
    else hi = x;
    const pdf = x > 0 && x < 1
      ? comps.reduce((s, c) => s + c.w * Math.exp((c.a - 1) * Math.log(x) + (c.b - 1) * Math.log1p(-x) - lbeta(c.a, c.b)), 0)
      : 0;
    let nxt = pdf > 0 ? x - cdf / pdf : NaN;
    if (!(lo < nxt && nxt < hi)) nxt = 0.5 * (lo + hi);
    if (Math.abs(nxt - x) < 1e-13) return nxt;
    x = nxt;
  }
  return x;
}

/**
 * Rates of several groups, each pulled towards the others as far as the data justify.
 * Full Bayes for the beta-binomial hierarchy with p(a, b) ∝ (a + b)^(-5/2) (Gelman et al.,
 * Bayesian Data Analysis, section 5.3), integrated on a grid.
 */
export function poolRates(k: number[], n: number[], level = 0.95): PooledRates {
  if (k.length !== n.length || k.length < 2) throw new RangeError("k and n need the same length, at least two groups");
  k.forEach((ki, i) => check(ki, n[i]));
  if (!k.some((ki, i) => ki > 0 && ki < n[i])) {
    throw new RangeError("at least one group needs both successes and failures (otherwise the posterior is improper)");
  }
  const g = grid(k, n);
  const mean = g.reduce((s, p) => s + (p.w * p.a) / (p.a + p.b), 0);
  let acc = 0;
  let medianV = g[0].v;
  for (const p of g.slice().sort((x, y) => x.v - y.v)) {
    acc += p.w;
    if (acc >= 0.5) {
      medianV = p.v;
      break;
    }
  }
  const tail = (1 - level) / 2;
  const estimates = k.map((ki, i) => {
    const comps = g.map((p) => ({ a: p.a + ki, b: p.b + n[i] - ki, w: p.w }));
    const m = comps.reduce((s, c) => s + (c.w * c.a) / (c.a + c.b), 0);
    return { estimate: m, lower: mixtureQuantile(tail, comps), upper: mixtureQuantile(1 - tail, comps) };
  });
  return { mean, concentration: Math.exp(medianV), raw: k.map((ki, i) => ki / n[i]), estimates };
}
