// Special functions and the optimiser shared by every model (spec sections 4, 5 and 7).

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** log Γ(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/** Regularised upper incomplete gamma Q(a, x). */
function gammaQ(a: number, x: number): number {
  if (x <= 0) return 1;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a;
    let sum = 1 / a;
    let del = sum;
    for (let n = 0; n < 10000; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-17) break;
    }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 10000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-17) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}

export function erfc(x: number): number {
  return x >= 0 ? gammaQ(0.5, x * x) : 2 - gammaQ(0.5, x * x);
}

/** Standard normal quantile, by bisection on Φ to full precision. */
export function normPpf(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError("p must be in (0, 1)");
  let lo = -40;
  let hi = 40;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;
    if (0.5 * erfc(-mid / Math.SQRT2) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** P(chi-squared with one degree of freedom > x). */
export function chisq1Sf(x: number): number {
  return erfc(Math.sqrt(Math.max(x, 0) / 2));
}

function betacf(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  d = 1 / (Math.abs(d) > tiny ? d : tiny);
  let h = d;
  for (let m = 1; m <= 10000; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    d = 1 / (Math.abs(d) > tiny ? d : tiny);
    c = 1 + aa / c;
    c = Math.abs(c) > tiny ? c : tiny;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    d = 1 / (Math.abs(d) > tiny ? d : tiny);
    c = 1 + aa / c;
    c = Math.abs(c) > tiny ? c : tiny;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) return h;
  }
  throw new Error("incomplete beta: continued fraction did not converge");
}

/** Regularised incomplete beta function I_x(a, b). */
export function betaCdf(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(a * Math.log(x) + b * Math.log1p(-x) - lbeta(a, b));
  if (x < (a + 1) / (a + b + 2)) return (front * betacf(a, b, x)) / a;
  return 1 - (front * betacf(b, a, 1 - x)) / b;
}

/** Quantile of Beta(a, b) by bisection to full double precision. */
export function betaPpf(p: number, a: number, b: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;
    if (betaCdf(mid, a, b) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// ---- small dense linear algebra ------------------------------------------------------------

export type Matrix = number[][];

export function inversePd(m: Matrix): Matrix | null {
  const n = m.length;
  const low: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = m[i][j];
      for (let k = 0; k < j; k++) s -= low[i][k] * low[j][k];
      if (i === j) {
        if (!(s > 0) || !Number.isFinite(s)) return null;
        low[i][i] = Math.sqrt(s);
      } else {
        low[i][j] = s / low[j][j];
      }
    }
  }
  const inv: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let col = 0; col < n; col++) {
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = i === col ? 1 : 0;
      for (let k = 0; k < i; k++) s -= low[i][k] * y[k];
      y[i] = s / low[i][i];
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= low[k][i] * x[k];
      x[i] = s / low[i][i];
    }
    for (let i = 0; i < n; i++) inv[i][col] = x[i];
  }
  return inv;
}

type Fn = (x: number[]) => number;

export function gradient(f: Fn, x: number[], h = 1e-5): number[] {
  return x.map((_, i) => {
    const up = x.slice();
    const dn = x.slice();
    up[i] += h;
    dn[i] -= h;
    return (f(up) - f(dn)) / (2 * h);
  });
}

export function hessian(f: Fn, x: number[], h = 1e-3): Matrix {
  const n = x.length;
  const out: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  const f0 = f(x);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let v: number;
      if (i === j) {
        const up = x.slice();
        const dn = x.slice();
        up[i] += h;
        dn[i] -= h;
        v = (f(up) - 2 * f0 + f(dn)) / (h * h);
      } else {
        const pp = x.slice();
        const pm = x.slice();
        const mp = x.slice();
        const mm = x.slice();
        pp[i] += h; pp[j] += h;
        pm[i] += h; pm[j] -= h;
        mp[i] -= h; mp[j] += h;
        mm[i] -= h; mm[j] -= h;
        v = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h);
      }
      out[i][j] = v;
      out[j][i] = v;
    }
  }
  return out;
}

// ---- optimiser (spec section 4) ------------------------------------------------------------

function nelderMead(f: Fn, x0: number[]): { x: number[]; value: number; iterations: number } {
  const n = x0.length;
  let pts: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) pts.push(x0.map((v, j) => v + (j === i ? 0.5 : 0)));
  let vals = pts.map(f);
  let it = 0;
  while (it < 5000) {
    const order = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
    pts = order.map((i) => pts[i]);
    vals = order.map((i) => vals[i]);
    const spread = vals[n] - vals[0];
    let size = 0;
    for (let i = 1; i <= n; i++) for (let j = 0; j < n; j++) size = Math.max(size, Math.abs(pts[i][j] - pts[0][j]));
    if (spread < 1e-10 && size < 1e-8) break;
    it++;
    const cen = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cen[j] += pts[i][j];
    for (let j = 0; j < n; j++) cen[j] /= n;
    const worst = pts[n];
    const xr = cen.map((c, j) => c + (c - worst[j]));
    const fr = f(xr);
    if (fr < vals[0]) {
      const xe = cen.map((c, j) => c + 2 * (c - worst[j]));
      const fe = f(xe);
      if (fe < fr) {
        pts[n] = xe;
        vals[n] = fe;
      } else {
        pts[n] = xr;
        vals[n] = fr;
      }
      continue;
    }
    if (fr < vals[n - 1]) {
      pts[n] = xr;
      vals[n] = fr;
      continue;
    }
    if (fr < vals[n]) {
      const xc = cen.map((c, j) => c + 0.5 * (xr[j] - c));
      const fc = f(xc);
      if (fc <= fr) {
        pts[n] = xc;
        vals[n] = fc;
        continue;
      }
    } else {
      const xc = cen.map((c, j) => c + 0.5 * (worst[j] - c));
      const fc = f(xc);
      if (fc < vals[n]) {
        pts[n] = xc;
        vals[n] = fc;
        continue;
      }
    }
    const best = pts[0];
    for (let i = 1; i <= n; i++) {
      pts[i] = best.map((b, j) => b + 0.5 * (pts[i][j] - b));
      vals[i] = f(pts[i]);
    }
  }
  let best = 0;
  for (let i = 1; i <= n; i++) if (vals[i] < vals[best]) best = i;
  return { x: pts[best], value: vals[best], iterations: it };
}

function safe(f: Fn): Fn {
  return (x) => {
    if (Math.max(...x.map(Math.abs)) > 20) return Infinity;
    let v: number;
    try {
      v = f(x);
    } catch {
      return Infinity;
    }
    return Number.isFinite(v) ? v : Infinity;
  };
}

export interface Minimum {
  x: number[];
  value: number;
  cov: Matrix | null;
  converged: boolean;
  iterations: number;
}

/** Nelder-Mead, then Newton polish; covariance from the Richardson-extrapolated Hessian. */
export function minimise(fn: Fn, x0: number[]): Minimum {
  const f = safe(fn);
  const nm = nelderMead(f, x0);
  let x = nm.x;
  let fx = nm.value;
  for (let k = 0; k < 20; k++) {
    const inv = inversePd(hessian(f, x));
    if (inv === null) break;
    const g = gradient(f, x);
    const step = x.map((_, i) => -inv[i].reduce((s, v, j) => s + v * g[j], 0));
    if (Math.max(...step.map(Math.abs)) < 1e-10) break;
    let scale = 1;
    let moved = false;
    for (let j = 0; j < 30; j++) {
      const cand = x.map((v, i) => v + scale * step[i]);
      const fc = f(cand);
      if (fc < fx) {
        x = cand;
        fx = fc;
        moved = true;
        break;
      }
      scale *= 0.5;
    }
    if (!moved) break;
  }
  const h1 = hessian(f, x, 1e-3);
  const h2 = hessian(f, x, 2e-3);
  const h = h1.map((row, i) => row.map((v, j) => (4 * v - h2[i][j]) / 3));
  const cov = inversePd(h);
  const g = gradient(f, x);
  return { x, value: fx, cov, converged: cov !== null && Math.max(...g.map(Math.abs)) < 1e-3, iterations: nm.iterations };
}
