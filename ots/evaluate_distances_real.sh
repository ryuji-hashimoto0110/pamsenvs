seed=42
ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/1min"
ticker_folder_names="3382 4568 6501 6502 7203 8306 8411 9202"
tickers="3382 4568 6501 6502 7203 8306 8411 9202"
resample_rule="1min"
point_cloud_type="return"
distance_matrix_save_path="distance_matrixes/distance_matrix_real.csv"
n_samples=600
fig_save_path="../imgs/distance_matrix_real.pdf"
python evaluate_distances_real.py \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_folder_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type} \
--distance_matrix_save_path ${distance_matrix_save_path} \
--n_samples ${n_samples} \
--fig_save_path ${fig_save_path}