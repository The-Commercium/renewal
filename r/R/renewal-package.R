#' renewal: retention, lifetime and conversion statistics for subscriptions
#'
#' Start with [fit_sbg()] or [fit_bdw()] for retention, then [survival()], [retention()],
#' [discounted_lifetime()] and [residual_lifetime()]. For conversion rates see
#' [rate_interval()] and [pool_rates()]. The full specification, shared with the Python and
#' TypeScript implementations, is in `spec/MODEL.md` of the source repository.
#'
#' @keywords internal
#' @importFrom stats coef vcov logLik nobs AIC BIC
"_PACKAGE"
