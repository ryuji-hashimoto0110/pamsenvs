ohlcv_file_path="../datas/real_datas/intraday/flex_ohlcv/all_time/9202_20150105_20210820.csv"
#ohlcv_file_path="../datas/artificial_datas/intraday/flex_ohlcv/all_time/asymmetric_volatility_2042/afcn_af30_20150101_20190209.csv"
obs_freq="1min"
obs_num2calc_return=300
close_name="close"
Rscript check_asymmetry.R ${ohlcv_file_path} ${obs_freq} ${obs_num2calc_return} ${close_name}