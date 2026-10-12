# renewal (TypeScript)

Retention, lifetime and conversion statistics for subscription products, in TypeScript with no
runtime dependencies. Runs in Node and in the browser. One of three implementations (R, Python,
TypeScript) of the same specification.

```ts
import { fitSbg, poolRates } from "renewal";

const fit = fitSbg({ alive: [1000, 631, 468, 382, 326, 289, 262, 241] });
console.log(fit.summary());          // alpha 0.704, beta 1.182 (Fader and Hardie 2007)
fit.survival(12);                     // { estimate, lower, upper }
fit.discountedLifetime(0.1);
poolRates([3, 10, 1, 25], [40, 90, 15, 200]);
```

Documentation, validation and the model specification: https://github.com/The-Commercium/renewal
