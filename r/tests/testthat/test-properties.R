fh_regular <- c(1000, 631, 468, 382, 326, 289, 262, 241, 223, 207, 194, 183, 173)

test_that("the published Fader and Hardie (2007) estimates come back", {
  f <- fit_sbg(alive = fh_regular[1:8])
  expect_equal(round(unname(coef(f)), 3), c(0.704, 1.182))
  g <- fit_sbg(alive = c(1000, 869, 743, 653, 593, 551, 517, 491))
  expect_equal(round(unname(coef(g)), 3), c(0.668, 3.806))
})

test_that("cohort counts and individual records give the same fit", {
  alive <- fh_regular[1:8]
  periods <- c(rep(1:7, times = -diff(alive)), rep(8, alive[8]))
  active <- c(rep(FALSE, sum(-diff(alive))), rep(TRUE, alive[8]))
  a <- fit_sbg(alive = alive)
  b <- fit_sbg(periods, active)
  expect_equal(a$loglik, b$loglik, tolerance = 1e-9)
})

test_that("standard generics work", {
  f <- fit_sbg(alive = fh_regular[1:8])
  expect_s3_class(logLik(f), "logLik")
  expect_equal(AIC(f), 2 * 2 - 2 * f$loglik)
  expect_equal(nobs(f), 1000)
  expect_equal(dim(vcov(f)), c(2, 2))
  expect_output(print(f), "shifted beta-geometric")
})

test_that("the BdW nests the sBG and the test is an htest", {
  b <- fit_bdw(alive = fh_regular[1:8])
  s <- fit_sbg(alive = fh_regular[1:8])
  expect_gte(b$loglik, s$loglik - 1e-9)
  ht <- compare_models(s, b)
  expect_s3_class(ht, "htest")
  expect_true(ht$p.value >= 0 && ht$p.value <= 1)
})

test_that("retention rises under the sBG and intervals contain estimates", {
  f <- fit_sbg(alive = fh_regular[1:8])
  r <- retention(f, 1:14)$estimate
  expect_true(all(diff(r) > 0))
  s <- survival(f, 5)
  expect_true(s$lower < s$estimate && s$estimate < s$upper)
  expect_true(is.infinite(expected_lifetime(f)$estimate))
})

test_that("bad input is refused", {
  expect_error(fit_sbg(alive = 100))
  expect_error(fit_sbg(alive = c(100, 120)))
  expect_error(fit_sbg(alive = c(100, 100, 100)))
  expect_error(fit_sbg(c(1, 1, 1), c(FALSE, FALSE, FALSE)))
  expect_error(rate_interval(5, 3))
  expect_error(discounted_lifetime(fit_bdw(alive = fh_regular[1:8]), 0))
})

test_that("pooling moves an extreme small group towards the others", {
  r <- pool_rates(c(1, 9, 30), c(20, 30, 100))
  expect_gt(r$groups$estimate[1], r$groups$raw[1])
  expect_true(all(r$groups$lower < r$groups$estimate & r$groups$estimate < r$groups$upper))
})
