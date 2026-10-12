# Contractual retention models: sBG and BdW (spec sections 2 to 6).

log_survival <- function(t, alpha, beta, c) {
  ifelse(t <= 0, 0, lbeta(alpha, beta + pmax(t, 0)^c) - lbeta(alpha, beta))
}

log_churn <- function(t, alpha, beta, c, model) {
  if (model == "sbg") return(lbeta(alpha + 1, beta + t - 1) - lbeta(alpha, beta))
  prev <- log_survival(t - 1, alpha, beta, c)
  prev + log(-expm1(log_survival(t, alpha, beta, c) - prev))
}

as_counts <- function(periods, active, alive) {
  if (!is.null(alive)) {
    if (!missing(periods) && !is.null(periods)) stop("give either periods and active, or alive, not both")
    a <- as.numeric(alive)
    if (length(a) < 2 || a[1] <= 0 || any(a < 0) || any(a != round(a))) {
      stop("alive needs at least two whole, non-negative counts and alive[1] > 0")
    }
    if (any(diff(a) > 0)) stop("alive must not increase")
    k <- length(a) - 1
    counts <- data.frame(t = seq_len(k + 1), dead = c(-diff(a), 0), cens = c(rep(0, k), a[k + 1]))
    n <- a[1]
  } else {
    if (is.null(periods) || is.null(active)) stop("give periods and active, or alive")
    if (length(periods) != length(active) || length(periods) == 0) {
      stop("periods and active must have the same, non-zero length")
    }
    if (any(periods != round(periods)) || any(periods < 1)) stop("periods must be whole numbers of at least 1")
    active <- as.logical(active)
    if (anyNA(active)) stop("active must be TRUE or FALSE")
    ts <- sort(unique(periods))
    counts <- data.frame(t = ts,
                         dead = vapply(ts, function(v) sum(periods == v & !active), numeric(1)),
                         cens = vapply(ts, function(v) sum(periods == v & active), numeric(1)))
    n <- length(periods)
  }
  counts <- counts[counts$dead > 0 | counts$cens > 0, , drop = FALSE]
  if (sum(counts$dead) == 0) stop("no cancellations observed: the model cannot be estimated")
  if (sum((counts$dead + counts$cens)[counts$t >= 2]) == 0) {
    stop("no renewals observed: the model cannot be estimated")
  }
  list(counts = counts, n = n)
}

retention_loglik <- function(alpha, beta, c, model, counts) {
  d <- counts$dead > 0
  s <- counts$cens > 0
  sum(counts$dead[d] * log_churn(counts$t[d], alpha, beta, c, model)) +
    sum(counts$cens[s] * log_survival(counts$t[s] - 1, alpha, beta, c))
}

#' Fit a contractual retention model
#'
#' `fit_sbg()` fits the shifted beta-geometric model of Fader and Hardie (2007);
#' `fit_bdw()` fits the beta-discrete-Weibull model of Fader et al. (2018), which adds
#' duration dependence `c`. Both are fitted by maximum likelihood on the log scale, with the
#' optimiser of the package specification (Nelder-Mead, then Newton steps).
#'
#' Give either individual records (`periods` paid so far, and whether each subscriber is still
#' `active`) or cohort survivor counts (`alive`: the size of the starting cohort, then how many
#' renewed once, twice, and so on). Active subscribers are right-censored: they count as having
#' renewed `periods - 1` times.
#'
#' @param periods Whole number of periods each subscriber has paid for (at least 1).
#' @param active `TRUE` if the subscriber has not cancelled.
#' @param alive Cohort survivor counts, starting with the cohort size.
#' @return An object of class `renewal_fit`, with methods for `print()`, `coef()`, `vcov()`,
#'   `logLik()` (so `AIC()` and `BIC()` work) and `nobs()`. Its `boundary` element is `TRUE`
#'   when alpha + beta > 1e5: the data cannot tell subscribers apart yet. Survival within the
#'   data remains usable; alpha, beta and long-run value (DEL, DERL) do not, because missing
#'   heterogeneity biases them low.
#' @references Fader, P. S. and Hardie, B. G. S. (2007). How to project customer retention.
#'   *Journal of Interactive Marketing* 21(1), 76-90.
#'
#'   Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J. and Steenburgh, T. (2018). "How to
#'   project customer retention" revisited: the role of duration dependence. *Journal of
#'   Interactive Marketing* 43, 1-16.
#' @examples
#' # Fader and Hardie (2007), "regular" customers, years 0 to 7
#' fit <- fit_sbg(alive = c(1000, 631, 468, 382, 326, 289, 262, 241))
#' fit
#' survival(fit, c(8, 12))
#' discounted_lifetime(fit, d = 0.1)
#' @export
fit_sbg <- function(periods = NULL, active = NULL, alive = NULL) {
  fit_model("sbg", periods, active, alive)
}

#' @rdname fit_sbg
#' @export
fit_bdw <- function(periods = NULL, active = NULL, alive = NULL) {
  fit_model("bdw", periods, active, alive)
}

fit_model <- function(model, periods, active, alive) {
  data <- as_counts(periods, active, alive)
  k <- if (model == "sbg") 2 else 3
  nll <- function(x) {
    c <- if (k == 3) exp(x[3]) else 1
    -retention_loglik(exp(x[1]), exp(x[2]), c, model, data$counts)
  }
  res <- minimise(nll, rep(0, k))
  structure(list(model = model, x = res$x, loglik = -res$value, vcov = res$cov, n = data$n,
                 converged = res$converged, iterations = res$iterations, counts = data$counts,
                 boundary = exp(res$x[1]) + exp(res$x[2]) > 1e5),
            class = "renewal_fit")
}

fit_par <- function(fit, x) {
  c(alpha = exp(x[1]), beta = exp(x[2]), c = if (length(x) == 3) exp(x[3]) else 1)
}

delta_interval <- function(fit, fn, transform, level) {
  value <- fn(fit$x)
  if (!is.finite(value) || is.null(fit$vcov)) return(c(value, NaN, NaN))
  if (transform == "logit") {
    fwd <- function(v) log(v) - log1p(-v)
    back <- function(u) 1 / (1 + exp(-u))
  } else {
    fwd <- log
    back <- exp
  }
  g <- tryCatch(num_gradient(function(x) fwd(fn(x)), fit$x), error = function(e) NaN)
  if (any(!is.finite(g))) return(c(value, NaN, NaN))
  se <- sqrt(max(as.numeric(t(g) %*% fit$vcov %*% g), 0))
  z <- stats::qnorm((1 + level) / 2)
  u <- fwd(value)
  c(value, back(u - z * se), back(u + z * se))
}

estimates_frame <- function(key, rows) {
  m <- matrix(unlist(rows), ncol = 3, byrow = TRUE)
  out <- data.frame(key, m[, 1], m[, 2], m[, 3])
  names(out) <- c(names(key), "estimate", "lower", "upper")
  out
}

check_whole <- function(v, low, name) {
  if (any(v != round(v)) || any(v < low)) stop(sprintf("%s must be whole numbers of at least %d", name, low))
}

#' Survival, churn and retention of a fitted model
#'
#' `survival()` is the share of a new cohort that renews `t` times, S(t).
#' `churn_probability()` is the probability that a new subscriber pays for exactly `t` periods.
#' `retention()` is, of those who paid for period `t`, the share who renew for period `t + 1`.
#' Intervals use the delta method on the logit scale.
#'
#' @param fit A `renewal_fit`.
#' @param t Whole numbers of periods.
#' @param level Interval level.
#' @return A data frame with columns `t`, `estimate`, `lower`, `upper`.
#' @examples
#' fit <- fit_sbg(alive = c(1000, 631, 468, 382, 326, 289, 262, 241))
#' retention(fit, 1:5)
#' @export
survival <- function(fit, t, level = 0.95) {
  check_whole(t, 0, "t")
  estimates_frame(list(t = t), lapply(t, function(tt) {
    if (tt == 0) return(c(1, 1, 1))
    delta_interval(fit, function(x) {
      p <- fit_par(fit, x)
      exp(log_survival(tt, p[1], p[2], p[3]))
    }, "logit", level)
  }))
}

#' @rdname survival
#' @export
churn_probability <- function(fit, t, level = 0.95) {
  check_whole(t, 1, "t")
  estimates_frame(list(t = t), lapply(t, function(tt) {
    delta_interval(fit, function(x) {
      p <- fit_par(fit, x)
      exp(log_churn(tt, p[1], p[2], p[3], fit$model))
    }, "logit", level)
  }))
}

#' @rdname survival
#' @export
retention <- function(fit, t, level = 0.95) {
  check_whole(t, 1, "t")
  estimates_frame(list(t = t), lapply(t, function(tt) {
    delta_interval(fit, function(x) {
      p <- fit_par(fit, x)
      exp(log_survival(tt, p[1], p[2], p[3]) - log_survival(tt - 1, p[1], p[2], p[3]))
    }, "logit", level)
  }))
}

residual_value <- function(model, a, b, c, d, n) {
  if (d < 0) stop("the discount rate d must be >= 0")
  if (d == 0) {
    if (model != "sbg") stop("the BdW needs a discount rate d > 0")
    if (a <= 1) return(Inf)
    return(if (n == 0) (a + b - 1) / (a - 1) else (b + n - 1) / (a - 1))
  }
  log_rho <- -log1p(d)
  base <- if (n >= 1) log_survival(n - 1, a, b, c) else 0
  total <- 0
  # vectorised blocks of terms; stop at the first term below 1e-15 of the running sum
  start <- 0
  while (start < 1e6) {
    t <- start:(start + 4095)
    terms <- exp(log_survival(n + t, a, b, c) - base + t * log_rho)
    run <- total + cumsum(terms)
    hit <- which(terms < 1e-15 * run)
    if (length(hit)) return(run[hit[1]])
    total <- run[length(run)]
    start <- start + 4096
  }
  stop("discounted lifetime did not converge within 10^6 terms")
}

#' Discounted expected lifetime and residual lifetime
#'
#' `discounted_lifetime()` (DEL) is the expected number of payments of a new subscriber,
#' discounted at rate `d` per period, the first payment undiscounted.
#' `residual_lifetime()` (DERL) is the expected discounted future payments of a subscriber who
#' has paid for `n` periods and is at the renewal point for period `n + 1`.
#' `expected_lifetime()` is the undiscounted expected number of payments (sBG only, infinite when
#' alpha <= 1). Multiply by the net revenue per period for a value in money.
#'
#' @param fit A `renewal_fit`.
#' @param d Discount rate per period (`d > 0` for the BdW).
#' @param n Periods paid so far (at least 1).
#' @param level Interval level.
#' @return A data frame with the inputs and `estimate`, `lower`, `upper` (delta method, log
#'   scale).
#' @references Fader, P. S. and Hardie, B. G. S. (2010). Customer-base valuation in a
#'   contractual setting: the perils of ignoring heterogeneity. *Marketing Science* 29(1),
#'   85-93.
#' @examples
#' fit <- fit_sbg(alive = c(1000, 631, 468, 382, 326, 289, 262, 241))
#' discounted_lifetime(fit, c(0.05, 0.1))
#' residual_lifetime(fit, d = 0.1, n = 1:4)
#' @export
discounted_lifetime <- function(fit, d, level = 0.95) {
  estimates_frame(list(d = d), lapply(d, function(dd) {
    delta_interval(fit, function(x) {
      p <- fit_par(fit, x)
      residual_value(fit$model, p[1], p[2], p[3], dd, 0)
    }, "log", level)
  }))
}

#' @rdname discounted_lifetime
#' @export
residual_lifetime <- function(fit, d, n, level = 0.95) {
  check_whole(n, 1, "n")
  grid <- expand.grid(d = d, n = n)
  estimates_frame(grid, lapply(seq_len(nrow(grid)), function(i) {
    delta_interval(fit, function(x) {
      p <- fit_par(fit, x)
      residual_value(fit$model, p[1], p[2], p[3], grid$d[i], grid$n[i])
    }, "log", level)
  }))
}

#' @rdname discounted_lifetime
#' @export
expected_lifetime <- function(fit, level = 0.95) {
  if (fit$model != "sbg") stop("the BdW has no closed-form expected lifetime; use discounted_lifetime(d > 0)")
  discounted_lifetime(fit, 0, level)[, -1]
}

#' Median lifetime and expected cohort size
#'
#' @param fit A `renewal_fit`.
#' @param start Cohort size.
#' @param periods Number of renewals to project.
#' @return `median_lifetime()`: the smallest `t` with S(t) <= 0.5. `expected_cohort()`: expected
#'   number still subscribed after 0, 1, ..., `periods` renewals.
#' @export
median_lifetime <- function(fit) {
  p <- fit_par(fit, fit$x)
  start <- 1
  while (start < 1e6) {
    t <- start:(start + 4095)
    hit <- which(log_survival(t, p[1], p[2], p[3]) <= log(0.5))
    if (length(hit)) return(t[hit[1]])
    start <- start + 4096
  }
  stop("median lifetime beyond 10^6 periods")
}

#' @rdname median_lifetime
#' @export
expected_cohort <- function(fit, start, periods) {
  p <- fit_par(fit, fit$x)
  start * exp(log_survival(0:periods, p[1], p[2], p[3]))
}

#' Survival curve for given parameters
#'
#' @param t Whole numbers of periods.
#' @param alpha,beta Beta distribution parameters.
#' @param c Duration dependence (1 for the sBG).
#' @return S(t).
#' @examples
#' survival_curve(0:12, alpha = 0.704, beta = 1.182)
#' @export
survival_curve <- function(t, alpha, beta, c = 1) {
  check_whole(t, 0, "t")
  exp(log_survival(t, alpha, beta, c))
}

#' Likelihood-ratio test of the sBG against the BdW
#'
#' Tests `c = 1` (no duration dependence) with one degree of freedom.
#'
#' @param sbg,bdw Fits of the two models to the same data.
#' @return An object of class `htest`.
#' @examples
#' alive <- c(1000, 869, 743, 653, 593, 551, 517, 491)
#' compare_models(fit_sbg(alive = alive), fit_bdw(alive = alive))
#' @export
compare_models <- function(sbg, bdw) {
  if (sbg$model != "sbg" || bdw$model != "bdw") stop("compare_models(sbg, bdw) expects an sBG fit then a BdW fit")
  if (!identical(sbg$counts, bdw$counts)) stop("both fits must use the same data")
  lr <- max(2 * (bdw$loglik - sbg$loglik), 0)
  structure(list(statistic = c(LR = lr), parameter = c(df = 1),
                 p.value = stats::pchisq(lr, 1, lower.tail = FALSE),
                 method = "Likelihood-ratio test: sBG (c = 1) against beta-discrete-Weibull",
                 data.name = "the same retention data",
                 estimate = c(c = exp(bdw$x[3]))),
            class = "htest")
}

#' @export
coef.renewal_fit <- function(object, ...) {
  p <- fit_par(object, object$x)
  if (object$model == "sbg") p[1:2] else p
}

#' @export
vcov.renewal_fit <- function(object, ...) {
  v <- object$vcov
  if (!is.null(v)) dimnames(v) <- rep(list(paste0("log_", names(coef(object)))), 2)
  v
}

#' @export
logLik.renewal_fit <- function(object, ...) {
  structure(object$loglik, df = length(object$x), nobs = object$n, class = "logLik")
}

#' @export
nobs.renewal_fit <- function(object, ...) object$n

#' Parameter estimates with intervals
#'
#' @param fit A `renewal_fit`.
#' @param level Interval level.
#' @return A data frame with `parameter`, `estimate`, `lower`, `upper` (log-scale Wald
#'   intervals).
#' @export
parameters <- function(fit, level = 0.95) {
  nm <- c("alpha", "beta", "c")[seq_along(fit$x)]
  estimates_frame(list(parameter = nm), lapply(seq_along(fit$x), function(i) {
    delta_interval(fit, function(x) exp(x[i]), "log", level)
  }))
}

#' @export
print.renewal_fit <- function(x, ...) {
  name <- if (x$model == "sbg") "shifted beta-geometric (sBG)" else "beta-discrete-Weibull (BdW)"
  cat(sprintf("%s, %d subscribers\n", name, as.integer(x$n)))
  cat(sprintf("log-likelihood %.4f, AIC %.2f, BIC %.2f, converged %s\n",
              x$loglik, stats::AIC(x), stats::BIC(x), x$converged))
  p <- parameters(x)
  for (i in seq_len(nrow(p))) {
    cat(sprintf("  %-5s %10.4f   [%.4f, %.4f]\n", p$parameter[i], p$estimate[i], p$lower[i], p$upper[i]))
  }
  if (x$boundary) {
    cat("  at the no-heterogeneity boundary (alpha + beta > 1e5): the data cannot tell subscribers\n",
        " apart yet. Survival within the data is usable; alpha, beta and long-run value (DEL, DERL)\n",
        " are not: missing heterogeneity biases them low.\n")
  }
  invisible(x)
}
