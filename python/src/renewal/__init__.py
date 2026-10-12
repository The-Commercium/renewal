"""renewal: retention, lifetime and conversion statistics for subscription products.

Shifted beta-geometric and beta-discrete-Weibull retention models with right-censored data,
discounted expected (residual) lifetime, and conversion rates with Jeffreys and
empirical-Bayes intervals. Pure Python, no dependencies.
"""

from .models import Estimate, RetentionFit, compare, fit_bdw, fit_sbg, survival_curve
from .rates import PooledRates, pool_rates, rate_interval

__all__ = [
    "Estimate",
    "PooledRates",
    "RetentionFit",
    "compare",
    "fit_bdw",
    "fit_sbg",
    "pool_rates",
    "rate_interval",
    "survival_curve",
]
__version__ = "0.1.0"
