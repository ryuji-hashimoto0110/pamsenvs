#install.packages(
#  c(
#    "Dict", "data.table", "e1071", "freqdom", "RcppProgress", "ggplot2", "ASV"
#  )
#)
library(Dict)
library(data.table)
library(e1071)
library(freqdom)
library(RcppProgress)
library(ggplot2)
library(ASV)
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

calc_return <- function(vprice, obs_num, is_percentage = TRUE) {
  mprice <- matrix(vprice, ncol = obs_num, byrow = TRUE)
  daily_vprice <- mprice[, obs_num]
  num_days <- length(daily_vprice)
  vlog_return <- log(
    daily_vprice[2: num_days]
  ) - log(
    daily_vprice[1: num_days - 1]
  )
  if (is_percentage) {
    vlog_return <- 100 * vlog_return
  }
  #vlog_return <- scale(vlog_return)
  return(vlog_return)
}

report_asv_mcmc <- function(
  vlog_return, sim_num = 5000, burn_num = 1000, seed = 42
) {
  set.seed(seed)
  asv_results <- try(
    asv_mcmc(
      vlog_return, nSim = sim_num, nBurn = burn_num
    )
  )
  if (typeof(asv_results) == "list") {
    vmu <- asv_results[[1]]
    vphi <- asv_results[[2]]
    vsigma_eta <- asv_results[[3]]
    vrho <- asv_results[[4]]
    ReportMCMC(
      cbind(vmu, vphi, vsigma_eta, vrho),
      vname = c(expression(mu), expression(phi),
                expression(sigma[eta]), expression(rho))
    )
  }
}

ohlcv_df <- fread(ohlcv_file_path)
vprice <- as.vector(ohlcv_df$close)
daily_obs_num <- freq_ohlcv_size_dic[obs_freq]
vlog_return <- calc_return(vprice, daily_obs_num)
report_asv_mcmc(vlog_return)