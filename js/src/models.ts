// Contractual retention models: shifted beta-geometric (sBG) and beta-discrete-Weibull (BdW).

import { chisq1Sf, gradient, lbeta, minimise, normPpf, type Matrix } from "./numeric.ts";

const MAX_TERMS = 1_000_000;

export type Model = "sbg" | "bdw";

/** A point estimate with the bounds of its interval. */
export interface Estimate {
  estimate: number;
  lower: number;
  upper: number;
}

// ---- distribution functions (spec section 2) -----------------------------------------------

export function logSurvival(t: number, alpha: number, beta: number, c: number): number {
  if (t <= 0) return 0;
  return lbeta(alpha, beta + t ** c) - lbeta(alpha, beta);
}

function logChurn(t: number, alpha: number, beta: number, c: number, model: Model): number {
  if (model === "sbg") return lbeta(alpha + 1, beta + t - 1) - lbeta(alpha, beta);
  const prev = logSurvival(t - 1, alpha, beta, c);
  return prev + Math.log(-Math.expm1(logSurvival(t, alpha, beta, c) - prev));
}

// ---- data (spec section 3) -----------------------------------------------------------------

/** Individual records, or cohort survivor counts starting with the cohort size. */
export type RetentionData = { periods: number[]; active: boolean[] } | { alive: number[] };

type Counts = Map<number, [number, number]>;

function toCounts(data: RetentionData): { counts: Counts; n: number } {
  const counts: Counts = new Map();
  const add = (t: number, slot: 0 | 1, v: number) => {
    const row = counts.get(t) ?? [0, 0];
    row[slot] += v;
    counts.set(t, row);
  };
  let n: number;
  if ("alive" in data) {
    const a = data.alive;
    if (a.length < 2 || !(a[0] > 0) || a.some((v) => v < 0 || !Number.isInteger(v))) {
      throw new RangeError("alive needs at least two whole, non-negative counts and alive[0] > 0");
    }
    for (let t = 1; t < a.length; t++) if (a[t] > a[t - 1]) throw new RangeError("alive must not increase");
    for (let t = 1; t < a.length; t++) if (a[t - 1] - a[t]) add(t, 0, a[t - 1] - a[t]);
    if (a[a.length - 1]) add(a.length, 1, a[a.length - 1]);
    n = a[0];
  } else {
    const { periods, active } = data;
    if (periods.length !== active.length || periods.length === 0) {
      throw new RangeError("periods and active must have the same, non-zero length");
    }
    periods.forEach((t, i) => {
      if (!Number.isInteger(t) || t < 1) throw new RangeError("periods must be whole numbers of at least 1");
      add(t, active[i] ? 1 : 0, 1);
    });
    n = periods.length;
  }
  let churned = 0;
  let renewed = 0;
  for (const [t, [dead, cens]] of counts) {
    churned += dead;
    if (t >= 2) renewed += dead + cens;
  }
  if (churned === 0) throw new RangeError("no cancellations observed: the model cannot be estimated");
  if (renewed === 0) throw new RangeError("no renewals observed: the model cannot be estimated");
  return { counts, n };
}

function logLik(alpha: number, beta: number, c: number, model: Model, counts: Counts): number {
  let ll = 0;
  for (const [t, [dead, cens]] of counts) {
    if (dead) ll += dead * logChurn(t, alpha, beta, c, model);
    if (cens) ll += cens * logSurvival(t - 1, alpha, beta, c);
  }
  return ll;
}

// ---- fitting (spec section 4) --------------------------------------------------------------

/** Fit the shifted beta-geometric model (Fader and Hardie 2007) by maximum likelihood. */
export function fitSbg(data: RetentionData): RetentionFit {
  return fit("sbg", data);
}

/** Fit the beta-discrete-Weibull model (Fader et al. 2018): the sBG with duration dependence c. */
export function fitBdw(data: RetentionData): RetentionFit {
  return fit("bdw", data);
}

function fit(model: Model, data: RetentionData): RetentionFit {
  const { counts, n } = toCounts(data);
  const k = model === "sbg" ? 2 : 3;
  const nll = (x: number[]) =>
    -logLik(Math.exp(x[0]), Math.exp(x[1]), k === 3 ? Math.exp(x[2]) : 1, model, counts);
  const res = minimise(nll, new Array(k).fill(0));
  return new RetentionFit(model, res.x, -res.value, res.cov, n, res.converged, res.iterations, counts);
}

function checkWhole(t: number, low: number, name = "t"): number {
  if (!Number.isInteger(t) || t < low) throw new RangeError(`${name} must be a whole number of at least ${low}`);
  return t;
}

/** A fitted sBG or BdW model. Every quantity comes with a delta-method interval. */
export class RetentionFit {
  readonly model: Model;
  readonly loglik: number;
  /** Covariance of (log alpha, log beta[, log c]). */
  readonly vcov: Matrix | null;
  readonly n: number;
  readonly converged: boolean;
  readonly iterations: number;
  readonly k: number;
  readonly aic: number;
  readonly bic: number;
  private readonly x: number[];
  private readonly counts: Counts;

  constructor(model: Model, x: number[], loglik: number, vcov: Matrix | null, n: number,
              converged: boolean, iterations: number, counts: Counts) {
    this.model = model;
    this.x = x;
    this.loglik = loglik;
    this.vcov = vcov;
    this.n = n;
    this.converged = converged;
    this.iterations = iterations;
    this.counts = counts;
    this.k = x.length;
    this.aic = 2 * this.k - 2 * loglik;
    this.bic = this.k * Math.log(n) - 2 * loglik;
  }

  get alpha(): number {
    return Math.exp(this.x[0]);
  }

  get beta(): number {
    return Math.exp(this.x[1]);
  }

  get c(): number {
    return this.k === 3 ? Math.exp(this.x[2]) : 1;
  }

  /** True when alpha + beta > 1e5: no detectable heterogeneity between subscribers. */
  get atBoundary(): boolean {
    return this.alpha + this.beta > 1e5;
  }

  /** @internal */
  sameData(other: RetentionFit): boolean {
    if (this.n !== other.n || this.counts.size !== other.counts.size) return false;
    for (const [t, row] of this.counts) {
      const o = other.counts.get(t);
      if (!o || o[0] !== row[0] || o[1] !== row[1]) return false;
    }
    return true;
  }

  private par(x: number[]): [number, number, number] {
    return [Math.exp(x[0]), Math.exp(x[1]), this.k === 3 ? Math.exp(x[2]) : 1];
  }

  private interval(fn: (x: number[]) => number, transform: "logit" | "log", level: number): Estimate {
    const value = fn(this.x);
    const nan = { estimate: value, lower: NaN, upper: NaN };
    if (!Number.isFinite(value) || this.vcov === null) return nan;
    const fwd = transform === "logit" ? (v: number) => Math.log(v) - Math.log1p(-v) : Math.log;
    const back = transform === "logit" ? (u: number) => 1 / (1 + Math.exp(-u)) : Math.exp;
    let g: number[];
    try {
      g = gradient((x) => fwd(fn(x)), this.x);
    } catch {
      return nan;
    }
    if (!g.every(Number.isFinite)) return nan;
    let v = 0;
    for (let i = 0; i < this.k; i++) for (let j = 0; j < this.k; j++) v += g[i] * this.vcov[i][j] * g[j];
    const z = normPpf((1 + level) / 2);
    const u = fwd(value);
    const se = Math.sqrt(Math.max(v, 0));
    return { estimate: value, lower: back(u - z * se), upper: back(u + z * se) };
  }

  /** Parameter estimates with log-scale Wald intervals. */
  params(level = 0.95): Record<string, Estimate> {
    const out: Record<string, Estimate> = {};
    ["alpha", "beta", "c"].slice(0, this.k).forEach((name, i) => {
      out[name] = this.interval((x) => Math.exp(x[i]), "log", level);
    });
    return out;
  }

  /** Share of a new cohort that renews t times, S(t). */
  survival(t: number, level = 0.95): Estimate {
    checkWhole(t, 0);
    if (t === 0) return { estimate: 1, lower: 1, upper: 1 };
    return this.interval((x) => Math.exp(logSurvival(t, ...this.par(x))), "logit", level);
  }

  /** Probability that a new subscriber pays for exactly t periods, P(T = t). */
  churnProbability(t: number, level = 0.95): Estimate {
    checkWhole(t, 1);
    return this.interval((x) => Math.exp(logChurn(t, ...this.par(x), this.model)), "logit", level);
  }

  /** Of those who paid for period t, the share who renew for period t + 1. */
  retention(t: number, level = 0.95): Estimate {
    checkWhole(t, 1);
    return this.interval((x) => {
      const [a, b, c] = this.par(x);
      return Math.exp(logSurvival(t, a, b, c) - logSurvival(t - 1, a, b, c));
    }, "logit", level);
  }

  /** Expected number of a cohort of `start` still subscribed after 0..periods renewals. */
  expectedCohort(start: number, periods: number): number[] {
    const [a, b, c] = this.par(this.x);
    return Array.from({ length: periods + 1 }, (_, t) => start * Math.exp(logSurvival(t, a, b, c)));
  }

  /** DEL: expected discounted payments of a new subscriber (first one undiscounted). */
  discountedLifetime(d: number, level = 0.95): Estimate {
    return this.interval((x) => residualValue(this.model, ...this.par(x), d, 0), "log", level);
  }

  /** DERL: expected discounted future payments of a subscriber who has paid n periods. */
  residualLifetime(d: number, n: number, level = 0.95): Estimate {
    checkWhole(n, 1, "n");
    return this.interval((x) => residualValue(this.model, ...this.par(x), d, n), "log", level);
  }

  /** E[T], the undiscounted expected number of payments (sBG only; infinite if alpha <= 1). */
  expectedLifetime(level = 0.95): Estimate {
    if (this.model !== "sbg") {
      throw new RangeError("the BdW has no closed-form expected lifetime; use discountedLifetime(d > 0)");
    }
    return this.discountedLifetime(0, level);
  }

  /** The smallest t with S(t) <= 0.5. */
  medianLifetime(): number {
    const [a, b, c] = this.par(this.x);
    for (let t = 1; t < MAX_TERMS; t++) if (logSurvival(t, a, b, c) <= Math.log(0.5)) return t;
    throw new Error("median lifetime beyond 10^6 periods");
  }

  summary(level = 0.95): string {
    const name = this.model === "sbg" ? "shifted beta-geometric (sBG)" : "beta-discrete-Weibull (BdW)";
    const lines = [
      `${name}, ${this.n} subscribers`,
      `log-likelihood ${this.loglik.toFixed(4)}, AIC ${this.aic.toFixed(2)}, BIC ${this.bic.toFixed(2)}, converged ${this.converged}`,
    ];
    for (const [key, e] of Object.entries(this.params(level))) {
      lines.push(`  ${key.padEnd(5)} ${e.estimate.toFixed(4).padStart(10)}   [${e.lower.toFixed(4)}, ${e.upper.toFixed(4)}]`);
    }
    if (this.atBoundary) {
      lines.push("  at the no-heterogeneity boundary (alpha + beta > 1e5): the data cannot tell subscribers apart yet. Survival within the data is usable; alpha, beta and long-run value (DEL, DERL) are not: missing heterogeneity biases them low.");
    }
    return lines.join("\n");
  }
}

/** DEL when n = 0, DERL(d, n) otherwise (spec 5.1, 5.2). */
export function residualValue(model: Model, a: number, b: number, c: number, d: number, n: number): number {
  if (d < 0) throw new RangeError("the discount rate d must be >= 0");
  if (d === 0) {
    if (model !== "sbg") throw new RangeError("the BdW needs a discount rate d > 0");
    if (a <= 1) return Infinity;
    return n === 0 ? (a + b - 1) / (a - 1) : (b + n - 1) / (a - 1);
  }
  const logRho = -Math.log1p(d);
  const base = n >= 1 ? logSurvival(n - 1, a, b, c) : 0;
  let total = 0;
  for (let t = 0; t < MAX_TERMS; t++) {
    const term = Math.exp(logSurvival(n + t, a, b, c) - base + t * logRho);
    total += term;
    if (term < 1e-15 * total) return total;
  }
  throw new Error("discounted lifetime did not converge within 10^6 terms");
}

/** Likelihood-ratio test of the sBG (c = 1) against the BdW, with AIC and BIC. */
export function compare(sbg: RetentionFit, bdw: RetentionFit) {
  if (sbg.model !== "sbg" || bdw.model !== "bdw") throw new RangeError("compare(sbg, bdw) expects an sBG fit then a BdW fit");
  if (!sbg.sameData(bdw)) throw new RangeError("both fits must use the same data");
  const lr = Math.max(2 * (bdw.loglik - sbg.loglik), 0);
  return { lr, df: 1, pValue: chisq1Sf(lr), aic: { sbg: sbg.aic, bdw: bdw.aic }, bic: { sbg: sbg.bic, bdw: bdw.bic } };
}

/** S(t) for given parameters (c = 1 is the sBG). */
export function survivalCurve(t: number, alpha: number, beta: number, c = 1): number {
  return Math.exp(logSurvival(checkWhole(t, 0), alpha, beta, c));
}
