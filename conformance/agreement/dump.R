# Every output of the R implementation on the conformance inputs, as JSON on stdout.
suppressMessages({ library(jsonlite) })
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
for (f in list.files(file.path(here, "..", "..", "r", "R"), full.names = TRUE)) source(f)
V <- fromJSON(file.path(here, "..", "vectors.json"), simplifyVector = FALSE)
out <- list()
for (i in seq_along(V$survival)) { c <- V$survival[[i]]; out[[sprintf("survival/%d", i - 1)]] <- survival_curve(c$t, c$alpha, c$beta, c$c) }
for (i in seq_along(V$residual)) { c <- V$residual[[i]]; out[[sprintf("residual/%d", i - 1)]] <- residual_value(c$model, c$alpha, c$beta, c$c, c$d, c$n) }
for (i in seq_along(V$rate_interval)) { c <- V$rate_interval[[i]]; e <- rate_interval(c$k, c$n); out[[sprintf("jeffreys/%d", i - 1)]] <- c(e$lower, e$upper) }
row3 <- function(df) c(df$estimate, df$lower, df$upper)
for (c in V$fits) {
  fn <- if (c$model == "sbg") fit_sbg else fit_bdw
  d <- c$data
  f <- if (!is.null(d$alive)) fn(alive = unlist(d$alive)) else fn(unlist(d$periods), unlist(d$active))
  k <- c$name
  p <- fit_par(f, f$x)
  out[[paste0(k, "/par")]] <- c(p[["alpha"]], p[["beta"]], p[["c"]], f$loglik)
  pr <- parameters(f)
  for (i in seq_len(nrow(pr))) out[[paste0(k, "/param/", pr$parameter[i])]] <- row3(pr[i, ])
  for (t in c(1, 3, 12)) out[[paste0(k, "/S/", t)]] <- row3(survival(f, t))
  for (t in c(1, 6)) out[[paste0(k, "/r/", t)]] <- row3(retention(f, t))
  for (dd in c(0.01, 0.1)) out[[paste0(k, "/DEL/", dd)]] <- row3(discounted_lifetime(f, dd))
  out[[paste0(k, "/DERL")]] <- row3(residual_lifetime(f, 0.1, 5))
}
for (i in seq_along(V$pool_rates)) {
  c <- V$pool_rates[[i]]
  r <- pool_rates(unlist(c$k), unlist(c$n))
  out[[sprintf("pooled/%d", i - 1)]] <- c(r$mean, r$concentration, as.vector(t(as.matrix(r$groups[, c("estimate", "lower", "upper")]))))
}
cat(toJSON(out, digits = NA, auto_unbox = TRUE))
