vectors <- function() {
  skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(test_path("vectors.json"), simplifyVector = FALSE)
}

close_rel <- function(got, want, rel) {
  if (is.infinite(want)) return(is.infinite(got))
  abs(got - want) <= rel * max(abs(want), 1e-300)
}

test_that("survival matches the 30-digit references", {
  v <- vectors()
  for (case in v$survival) {
    got <- survival_curve(case$t, case$alpha, case$beta, case$c)
    expect_true(close_rel(got, case$value, v$tolerances$closed_form_rel), info = paste(unlist(case), collapse = " "))
  }
})

test_that("DEL and DERL match the 30-digit references", {
  v <- vectors()
  for (case in v$residual) {
    got <- renewal:::residual_value(case$model, case$alpha, case$beta, case$c, case$d, case$n)
    expect_true(close_rel(got, case$value, v$tolerances$closed_form_rel), info = paste(unlist(case), collapse = " "))
  }
})

test_that("Jeffreys intervals match", {
  v <- vectors()
  for (case in v$rate_interval) {
    got <- rate_interval(case$k, case$n, case$level)
    expect_equal(got$estimate, case$estimate)
    expect_lte(abs(got$lower - case$lower), v$tolerances$jeffreys_abs)
    expect_lte(abs(got$upper - case$upper), v$tolerances$jeffreys_abs)
  }
})

test_that("fits and every interval match the references", {
  v <- vectors()
  tol <- v$tolerances
  for (case in v$fits) {
    fn <- if (case$model == "sbg") fit_sbg else fit_bdw
    d <- case$data
    f <- if (!is.null(d$alive)) fn(alive = unlist(d$alive)) else fn(unlist(d$periods), unlist(d$active))
    expect_true(f$converged, info = case$name)
    p <- renewal:::fit_par(f, f$x)
    for (key in c("alpha", "beta", "c")) {
      expect_true(close_rel(p[[key]], case$expected[[key]], tol$fit_param_rel), info = paste(case$name, key))
    }
    expect_lte(abs(f$loglik - case$expected$loglik), tol$fit_loglik_abs)
    check <- function(row, want, what) {
      got <- c(row$estimate, row$lower, row$upper)
      ok <- mapply(close_rel, got, unlist(want), MoreArgs = list(rel = tol$interval_rel))
      expect_true(all(ok), info = paste(case$name, what, paste(got, collapse = " "), paste(unlist(want), collapse = " ")))
    }
    q <- case$intervals
    pars <- parameters(f)
    for (nm in names(q$params)) check(pars[pars$parameter == nm, ], q$params[[nm]], nm)
    for (t in names(q$survival)) check(survival(f, as.numeric(t)), q$survival[[t]], paste("S", t))
    for (t in names(q$churn_probability)) check(churn_probability(f, as.numeric(t)), q$churn_probability[[t]], paste("P", t))
    for (t in names(q$retention)) check(retention(f, as.numeric(t)), q$retention[[t]], paste("r", t))
    for (dd in names(q$discounted_lifetime)) check(discounted_lifetime(f, as.numeric(dd)), q$discounted_lifetime[[dd]], paste("DEL", dd))
    for (key in names(q$residual_lifetime)) {
      parts <- as.numeric(strsplit(key, "|", fixed = TRUE)[[1]])
      check(residual_lifetime(f, parts[1], parts[2]), q$residual_lifetime[[key]], paste("DERL", key))
    }
  }
})

test_that("pooled rates match the reference implementation", {
  v <- vectors()
  rel <- v$tolerances$pooled_rel
  for (case in v$pool_rates) {
    r <- pool_rates(unlist(case$k), unlist(case$n), case$level)
    expect_true(close_rel(r$mean, case$mean, rel))
    expect_true(close_rel(r$concentration, case$concentration, rel))
    for (i in seq_along(case$estimates)) {
      got <- unlist(r$groups[i, c("estimate", "lower", "upper")])
      ok <- mapply(close_rel, got, unlist(case$estimates[[i]]), MoreArgs = list(rel = rel))
      expect_true(all(ok), info = paste(got, collapse = " "))
    }
  }
})
