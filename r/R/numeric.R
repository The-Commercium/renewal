# Optimiser and finite differences shared by every model (spec section 4).

safe_fn <- function(f) {
  force(f)
  function(x) {
    if (max(abs(x)) > 20) return(Inf)
    v <- tryCatch(suppressWarnings(f(x)), error = function(e) Inf)
    if (!is.finite(v)) Inf else v
  }
}

num_gradient <- function(f, x, h = 1e-5) {
  vapply(seq_along(x), function(i) {
    up <- x
    dn <- x
    up[i] <- up[i] + h
    dn[i] <- dn[i] - h
    (f(up) - f(dn)) / (2 * h)
  }, numeric(1))
}

num_hessian <- function(f, x, h = 1e-3) {
  n <- length(x)
  out <- matrix(0, n, n)
  f0 <- f(x)
  for (i in seq_len(n)) {
    for (j in i:n) {
      if (i == j) {
        up <- x
        dn <- x
        up[i] <- up[i] + h
        dn[i] <- dn[i] - h
        v <- (f(up) - 2 * f0 + f(dn)) / (h * h)
      } else {
        pp <- x; pm <- x; mp <- x; mm <- x
        pp[i] <- pp[i] + h; pp[j] <- pp[j] + h
        pm[i] <- pm[i] + h; pm[j] <- pm[j] - h
        mp[i] <- mp[i] - h; mp[j] <- mp[j] + h
        mm[i] <- mm[i] - h; mm[j] <- mm[j] - h
        v <- (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h)
      }
      out[i, j] <- v
      out[j, i] <- v
    }
  }
  out
}

inverse_pd <- function(m) {
  if (any(!is.finite(m))) return(NULL)
  ch <- tryCatch(chol(m), error = function(e) NULL)
  if (is.null(ch)) NULL else chol2inv(ch)
}

nelder_mead <- function(f, x0) {
  n <- length(x0)
  pts <- rbind(x0, t(vapply(seq_len(n), function(i) {
    p <- x0
    p[i] <- p[i] + 0.5
    p
  }, numeric(n))))
  if (n == 1) pts <- matrix(pts, ncol = 1)
  vals <- apply(pts, 1, f)
  it <- 0
  while (it < 5000) {
    o <- order(vals)
    pts <- pts[o, , drop = FALSE]
    vals <- vals[o]
    spread <- vals[n + 1] - vals[1]
    size <- max(abs(sweep(pts[-1, , drop = FALSE], 2, pts[1, ])))
    if (spread < 1e-10 && size < 1e-8) break
    it <- it + 1
    cen <- colMeans(pts[seq_len(n), , drop = FALSE])
    worst <- pts[n + 1, ]
    xr <- cen + (cen - worst)
    fr <- f(xr)
    if (fr < vals[1]) {
      xe <- cen + 2 * (cen - worst)
      fe <- f(xe)
      if (fe < fr) {
        pts[n + 1, ] <- xe
        vals[n + 1] <- fe
      } else {
        pts[n + 1, ] <- xr
        vals[n + 1] <- fr
      }
      next
    }
    if (fr < vals[n]) {
      pts[n + 1, ] <- xr
      vals[n + 1] <- fr
      next
    }
    if (fr < vals[n + 1]) {
      xc <- cen + 0.5 * (xr - cen)
      fc <- f(xc)
      if (fc <= fr) {
        pts[n + 1, ] <- xc
        vals[n + 1] <- fc
        next
      }
    } else {
      xc <- cen + 0.5 * (worst - cen)
      fc <- f(xc)
      if (fc < vals[n + 1]) {
        pts[n + 1, ] <- xc
        vals[n + 1] <- fc
        next
      }
    }
    best <- pts[1, ]
    for (i in 2:(n + 1)) {
      pts[i, ] <- best + 0.5 * (pts[i, ] - best)
      vals[i] <- f(pts[i, ])
    }
  }
  i <- which.min(vals)
  list(x = unname(pts[i, ]), value = unname(vals[i]), iterations = it)
}

minimise <- function(f, x0) {
  f <- safe_fn(f)
  nm <- nelder_mead(f, x0)
  x <- nm$x
  fx <- nm$value
  for (k in seq_len(20)) {
    inv <- inverse_pd(num_hessian(f, x))
    if (is.null(inv)) break
    step <- -as.vector(inv %*% num_gradient(f, x))
    if (max(abs(step)) < 1e-10) break
    scale <- 1
    moved <- FALSE
    for (j in seq_len(30)) {
      cand <- x + scale * step
      fc <- f(cand)
      if (fc < fx) {
        x <- cand
        fx <- fc
        moved <- TRUE
        break
      }
      scale <- scale * 0.5
    }
    if (!moved) break
  }
  h <- (4 * num_hessian(f, x, 1e-3) - num_hessian(f, x, 2e-3)) / 3
  cov <- inverse_pd(h)
  g <- num_gradient(f, x)
  list(x = x, value = fx, cov = cov,
       converged = !is.null(cov) && max(abs(g)) < 1e-3, iterations = nm$iterations)
}
