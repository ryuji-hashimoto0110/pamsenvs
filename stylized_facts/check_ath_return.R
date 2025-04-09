dir.create("~/R/libs", recursive = TRUE, showWarnings = FALSE)
.libPaths(c("~/R/libs", .libPaths()))
install.packages("data.table")
install.packages("Dict")
install.packages("dplyr")
library(data.table)
library(Dict)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
ohlcv_file_path <- args[1]
obs_freq <- args[2]
close_name <- args[4]

freq_ohlcv_size_dic <- Dict$new(
  "1s" = 18001,
  "10s" = 1801,
  "30s" = 601,
  "1min" = 301,
  "1min_bybit" = 1440,
  "5min" = 61,
  "15min" = 21
)

get_daily_prices <- function(vprice, obs_num) {
  mprice <- matrix(vprice, ncol = obs_num, byrow = TRUE)
  daily_vprice <- mprice[, obs_num]
  return(daily_vprice)
}

report_multiple_ols <- function(data, y_vars, x_var = "ath_nearness") {
  for (y in y_vars) {
    cat("\n============================\n")
    cat(sprintf("model: %s ~ %s\n", y, x_var))
    cat("============================\n")
    df <- data %>% filter(!is.na(.data[[x_var]]), !is.na(.data[[y]]))
    formula <- as.formula(paste(y, "~", x_var))
    model <- lm(formula, data = df)
    summary_model <- summary(model)
    print(summary_model$coefficients)
    cat(sprintf("R-squared: %.4f\n", summary_model$r.squared))
    cat(sprintf("Adjusted R-squared: %.4f\n", summary_model$adj.r.squared))
  }
}

ohlcv_df <- fread(ohlcv_file_path)
vprice <- as.vector(ohlcv_df$close)
daily_obs_num <- freq_ohlcv_size_dic[obs_freq]
daily_vprice <- get_daily_prices(vprice, daily_obs_num)
ols_df <- tibble(
  date = seq_along(daily_vprice),
  ath_nearness = daily_vprice / cummax(daily_vprice),
  future_return_1d = lead(daily_vprice, 1) / daily_vprice,
  future_return_5d = lead(daily_vprice, 5) / daily_vprice,
  future_return_10d = lead(daily_vprice, 10) / daily_vprice,
  future_return_15d = lead(daily_vprice, 15) / daily_vprice,
  future_return_30d = lead(daily_vprice, 30) / daily_vprice,
)
y_vars <- c(
  "future_return_1d",
  "future_return_5d",
  "future_return_10d",
  "future_return_15d",
  "future_return_30d"
)
report_multiple_ols(ols_df, y_vars, "ath_nearness")
