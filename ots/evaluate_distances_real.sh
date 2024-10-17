seed=42
ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/1min"
#ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/all_time"
ticker_folder_names="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
#ticker_file_names="3382_20150105_20210820.csv "\
#"4568_20150105_20210820.csv "\
#"6501_20150105_20210820.csv "\
#"6502_20150105_20210820.csv "\
#"7203_20150105_20210820.csv "\
#"8306_20150105_20210820.csv "\
#"8411_20150105_20210820.csv "\
#"9202_20150105_20210820.csv"
tickers="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
resample_rule="1min"
point_cloud_type="return_ts"
distance_matrix_save_path="distance_matrices/distance_matrix_return_ts_real.csv"
n_samples=1000
lags="1 10 20 30 50 100"
figs_save_path="../imgs/ots/return_ts/real"
nrows_subplots=3
ncols_subplots=6
python evaluate_distances_real.py \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_folder_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type} \
--distance_matrix_save_path ${distance_matrix_save_path} \
--n_samples ${n_samples} \
--lags ${lags} \
--figs_save_path ${figs_save_path} \
--nrows_subolots ${nrows_subplots} \
--ncols_subplots ${ncols_subplots}