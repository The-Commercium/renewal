# renewal (Python)

Retention, lifetime and conversion statistics for subscription products, in pure Python with no
dependencies. One of three implementations (R, Python, TypeScript) of the same specification.

```python
import renewal as rn

fit = rn.fit_sbg(alive=[1000, 631, 468, 382, 326, 289, 262, 241])
print(fit)                              # alpha 0.704, beta 1.182 (Fader and Hardie 2007)
fit.survival(12)                        # share still subscribed after 12 renewals, with interval
fit.discounted_lifetime(0.1)            # expected discounted payments of a new subscriber
rn.pool_rates([3, 10, 1, 25], [40, 90, 15, 200])
```

Documentation, validation and the model specification: https://github.com/The-Commercium/renewal
