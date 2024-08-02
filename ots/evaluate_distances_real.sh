seed=42
ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/1min"
#ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/all_time"
ticker_folder_names="3382 4568 6501 6502 7203 8306 8411 9202"
#ticker_file_names="3382_20150105_20210820.csv "\
#"4568_20150105_20210820.csv "\
#"6501_20150105_20210820.csv "\
#"6502_20150105_20210820.csv "\
#"7203_20150105_20210820.csv "\
#"8306_20150105_20210820.csv "\
#"8411_20150105_20210820.csv "\
#"9202_20150105_20210820.csv"
tickers="3382 4568 6501 6502 7203 8306 8411 9202"
resample_rule="1min"
point_cloud_type="returns"
distance_matrix_save_path="distance_matrices/distance_matrix_returns_real.csv"
n_samples=1000
figs_save_path="../imgs/ots/returns/real"
python evaluate_distances_real.py \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_folder_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type} \
--distance_matrix_save_path ${distance_matrix_save_path} \
--n_samples ${n_samples} \
--figs_save_path ${figs_save_path}