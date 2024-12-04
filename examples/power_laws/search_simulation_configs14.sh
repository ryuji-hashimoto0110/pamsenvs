seed=42
ohlcv_folder_path="../../datas/real_datas/intraday/flex_ohlcv/1min"
ticker_folder_names="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
tickers="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
resample_rule="1min"
point_cloud_type="tail_return"

initial_seed=42
base_config_path="base_config14.json"
target_variables_config_path="target_variables_config14.json"
temp_txts_path="../../datas/artificial_datas/flex_txt/temp14"
temp_tick_dfs_path="../../datas/artificial_datas/flex_csv/temp14"
temp_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/temp14"
temp_all_time_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/all_time/temp14"
path_to_calc_point_clouds="../../datas/artificial_datas/intraday/flex_ohlcv/1min/temp14"
num_simulations=100
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
num_points=1500
results_save_path="ot_result14.csv"

python ../search_simulation_configs.py \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_folder_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type} \
--initial_seed ${initial_seed} \
--base_config_path ${base_config_path} \
--target_variables_config_path ${target_variables_config_path} \
--temp_txts_path ${temp_txts_path} \
--temp_tick_dfs_path ${temp_tick_dfs_path} \
--temp_ohlcv_dfs_path ${temp_ohlcv_dfs_path} \
--temp_all_time_ohlcv_dfs_path ${temp_all_time_ohlcv_dfs_path} \
--path_to_calc_point_clouds ${path_to_calc_point_clouds} \
--num_simulations ${num_simulations} \
--transactions_path ${transactions_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--num_points ${num_points} \
--results_save_path ${results_save_path} \
--show_process
