ohlcv_file_path="../datas/real_datas/intraday/flex_ohlcv/all_time/3407_20150101_20211231.csv"
obs_freq="1min"
close_name="close"
Rscript check_asymmetry.R ${ohlcv_file_path} ${obs_freq} ${close_name}