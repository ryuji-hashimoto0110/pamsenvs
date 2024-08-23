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
obs_num2calc_return <- as.integer(args[3])
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

calc_return <- function(
  vprice, daily_obs_num, obs_num2calc_return, is_percentage = TRUE
) {
  mprice <- matrix(vprice, ncol = daily_obs_num, byrow = TRUE)
  mreturn <-
    log(mprice[, 2: daily_obs_num]) - log(mprice[, 1: daily_obs_num - 1])
  vreturn <- as.vector(t(mreturn))
  mreturn_expanded <- matrix(vreturn, ncol = obs_num2calc_return, byrow = TRUE)
  vreturn_expanded <- apply(mreturn_expanded, 1, sum)
  if (is_percentage) {
    vreturn_expanded <- 100 * vreturn_expanded
  }
  #vreturn_expanded <- scale(vreturn_expanded)
  return(vreturn_expanded)
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
vlog_return <- calc_return(vprice, daily_obs_num, obs_num2calc_return)
report_asv_mcmc(vlog_return)