# Figure for the README: Fader and Hardie (2007) data, sBG fitted to years 0-7, hold-out 8-12.
#   Rscript validation/fader-hardie-2007/figure.R
for (f in list.files("r/R", full.names = TRUE)) source(f)
data <- list(
  regular = c(1000, 631, 468, 382, 326, 289, 262, 241, 223, 207, 194, 183, 173),
  `high end` = c(1000, 869, 743, 653, 593, 551, 517, 491, 468, 445, 427, 409, 394))
cols <- c(regular = "#2F5D8A", `high end` = "#B5532F")
png("docs/fader-hardie-2007.png", width = 1600, height = 1000, res = 200)
par(mar = c(4.2, 4.6, 2.6, 1), family = "sans", las = 1, bty = "l")
plot(NA, xlim = c(0, 12), ylim = c(0, 1), xlab = "renewals (years)", ylab = "share still subscribed",
     main = "sBG fitted to years 0 to 7; years 8 to 12 held out", font.main = 1, cex.main = 1)
rect(7, -1, 13, 2, col = "#F2F2F2", border = NA)
text(9.9, 0.97, "hold-out", col = "grey40", cex = 0.85)
for (seg in names(data)) {
  fit <- fit_sbg(alive = data[[seg]][1:8])
  s <- survival(fit, 0:12)
  polygon(c(s$t, rev(s$t)), c(s$lower, rev(s$upper)), col = adjustcolor(cols[[seg]], 0.18), border = NA)
  lines(s$t, s$estimate, col = cols[[seg]], lwd = 2)
  points(0:12, data[[seg]] / 1000, pch = 19, cex = 0.8, col = cols[[seg]])
}
legend("bottomleft", inset = c(0.02, 0.04), bty = "n", cex = 0.85,
       legend = c("regular: actual", "high end: actual", "fit with 95% interval"),
       col = c(cols, "grey30"), pch = c(19, 19, NA), lwd = c(NA, NA, 2))
invisible(dev.off())
