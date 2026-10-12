# Conversion rates with honest intervals for small counts (spec section 7).

check_counts <- function(k, n) {
  if (length(k) != length(n) || any(k != round(k)) || any(n != round(n)) || any(n < 1) ||
      any(k < 0) || any(k > n)) {
    stop("need whole numbers with 0 <= k <= n and n >= 1")
  }
}

#' A rate with its Jeffreys interval
#'
#' `k` successes in `n` trials (for example trials that became paid subscriptions). The
#' interval is the Jeffreys interval of Brown, Cai and DasGupta (2001).
#'
#' @param k Successes.
#' @param n Trials.
#' @param level Interval level.
#' @return A data frame with `k`, `n`, `estimate`, `lower`, `upper`.
#' @references Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a
#'   binomial proportion. *Statistical Science* 16(2), 101-133.
#' @examples
#' rate_interval(c(0, 3, 39), c(12, 40, 345))
#' @export
rate_interval <- function(k, n, level = 0.95) {
  check_counts(k, n)
  tail <- (1 - level) / 2
  lower <- ifelse(k == 0, 0, stats::qbeta(tail, k + 0.5, n - k + 0.5))
  upper <- ifelse(k == n, 1, stats::qbeta(1 - tail, k + 0.5, n - k + 0.5))
  data.frame(k = k, n = n, estimate = k / n, lower = lower, upper = upper)
}

log_hyper <- function(u, v, k, n) {
  mu <- 1 / (1 + exp(-u))
  kappa <- exp(v)
  a <- mu * kappa
  b <- (1 - mu) * kappa
  log(mu) + log1p(-mu) - v / 2 + sum(lbeta(a + k, b + n - k)) - length(k) * lbeta(a, b)
}

hyper_grid <- function(k, n) {
  lin <- function(lo, hi, num) lo + (hi - lo) * (seq_len(num) - 1) / (num - 1)
  coarse <- expand.grid(v = lin(-30, 20, 101), u = lin(-10, 10, 41))
  coarse$lp <- mapply(log_hyper, coarse$u, coarse$v, MoreArgs = list(k = k, n = n))
  keep <- coarse[coarse$lp > max(coarse$lp) - 20, ]
  fine <- expand.grid(v = lin(max(min(keep$v) - 0.5, -30), min(max(keep$v) + 0.5, 20), 101),
                      u = lin(max(min(keep$u) - 0.5, -10), min(max(keep$u) + 0.5, 10), 101))
  fine$lp <- mapply(log_hyper, fine$u, fine$v, MoreArgs = list(k = k, n = n))
  fine$w <- exp(fine$lp - max(fine$lp))
  fine <- fine[order(-fine$w), ]
  cum <- cumsum(fine$w)
  m <- which(cum >= (1 - 1e-10) * sum(fine$w))[1]
  fine <- fine[seq_len(m), ]
  mu <- 1 / (1 + exp(-fine$u))
  data.frame(a = mu * exp(fine$v), b = (1 - mu) * exp(fine$v), w = fine$w / cum[m], v = fine$v)
}

mixture_quantile <- function(q, a, b, w) {
  lo <- 0
  hi <- 1
  x <- sum(w * a / (a + b))
  for (i in seq_len(200)) {
    cdf <- sum(w * stats::pbeta(x, a, b)) - q
    if (abs(cdf) < 1e-14) return(x)
    if (cdf < 0) lo <- x else hi <- x
    pdf <- if (x > 0 && x < 1) sum(w * stats::dbeta(x, a, b)) else 0
    nxt <- if (pdf > 0) x - cdf / pdf else NaN
    if (!is.finite(nxt) || nxt <= lo || nxt >= hi) nxt <- 0.5 * (lo + hi)
    if (abs(nxt - x) < 1e-13) return(nxt)
    x <- nxt
  }
  x
}

#' Rates of several groups, pooled as far as the data justify
#'
#' A hierarchical beta-binomial model: each group's true rate comes from a shared Beta(a, b)
#' population with the hyperprior p(a, b) proportional to (a + b)^(-5/2) (Gelman et al., *Bayesian
#' Data Analysis*, 3rd ed., section 5.3). The hyperparameters are integrated out on a grid, so
#' the intervals include the uncertainty about how alike the groups are. Small groups move
#' towards the others; large groups barely move. At least one group must have both successes
#' and failures, otherwise the posterior is improper.
#'
#' @param k Successes per group.
#' @param n Trials per group.
#' @param level Interval level.
#' @param labels Optional group names.
#' @return An object of class `renewal_pooled`: a list with `groups` (a data frame with the raw
#'   rate, the pooled `estimate`, `lower`, `upper`), `mean` (posterior mean of the population
#'   rate) and `concentration` (posterior median of a + b).
#' @examples
#' pool_rates(k = c(3, 10, 1, 25), n = c(40, 90, 15, 200), labels = c("UK", "US", "JP", "DE"))
#' @export
pool_rates <- function(k, n, level = 0.95, labels = NULL) {
  check_counts(k, n)
  if (length(k) < 2) stop("need at least two groups")
  if (!any(k > 0 & k < n)) stop("at least one group needs both successes and failures (otherwise the posterior is improper)")
  g <- hyper_grid(k, n)
  ord <- order(g$v)
  median_v <- g$v[ord][which(cumsum(g$w[ord]) >= 0.5)[1]]
  tail <- (1 - level) / 2
  rows <- lapply(seq_along(k), function(i) {
    a <- g$a + k[i]
    b <- g$b + n[i] - k[i]
    c(sum(g$w * a / (a + b)), mixture_quantile(tail, a, b, g$w), mixture_quantile(1 - tail, a, b, g$w))
  })
  m <- matrix(unlist(rows), ncol = 3, byrow = TRUE)
  groups <- data.frame(group = if (is.null(labels)) seq_along(k) else labels, k = k, n = n,
                       raw = k / n, estimate = m[, 1], lower = m[, 2], upper = m[, 3])
  structure(list(groups = groups, mean = sum(g$w * g$a / (g$a + g$b)), concentration = exp(median_v),
                 level = level), class = "renewal_pooled")
}

#' @export
print.renewal_pooled <- function(x, ...) {
  cat(sprintf("Pooled rates: population mean %.4f, concentration %.2f\n", x$mean, x$concentration))
  print(x$groups, row.names = FALSE, digits = 4)
  invisible(x)
}
