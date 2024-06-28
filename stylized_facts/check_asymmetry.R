install.packages(
  c(
    Dict, data.table, e1071, freqdom, RcppProgress, ggplot2, ASV
  )
)
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

freq_ohlcv_size_dic <- Dict$new(
  "1s" = 18002,
  "10s" = 1802,
  "30s" = 602,
  "1min" = 302,
  "5min" = 62,
  "15min" = 22
)

calc_return <- function(vprice, obs_num, is_percentage = TRUE) {
  mprice <- matrix(vprice, ncol = obs_num, byrow = TRUE)
  vlog_return <- log(mprice[, obs_num]) - log(mprice[, 1])
  if (is_percentage) {
    vlog_return <- 100 * vlog_return
  }
  return(vlog_return)
}

report_asv_mcmc <- function(
  vlog_return,
  sim_num = 5000, burn_num = 1000, seed = 42
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
      cbind(vmu, vphi, vsigma_eta, vrho)
    )
  }
}

ohlcv_df <- fread(ohlcv_file_path)
vclose_name <- c("close", "Close", "CLOSE")
vclose_name <- vclose_name[! vclose_name %in% names(ohlcv_df)]
vprice <- ohlcv_df[, vclose_name]
ohlcv_size <- freq_ohlcv_size_dic[obs_freq]
vlog_return <- calc_return(vprice, obs_num)
report_mcmc(vlog_return)