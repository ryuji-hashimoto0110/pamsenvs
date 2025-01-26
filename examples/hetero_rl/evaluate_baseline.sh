initial_seed=42
significant_figures=10
config_path="base_config_fcn.json"
specific_name="fcn"
txts_path="../../datas/artificial_datas/flex_txt/hetero_rl/temp"
num_simulations=300
txts_save_path="../../datas/artificial_datas/flex_txt/hetero_rl/temp"
tick_dfs_save_path="../../datas/artificial_datas/flex_csv/hetero_rl/temp"
ohlcv_dfs_save_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/hetero_rl/temp"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_folder_path="../../imgs/hetero_rl"
results_save_path="../../stylized_facts/results/hetero_rl/baseline.csv"
ohlcv_folder_path="../../datas/real_datas/intraday/flex_ohlcv/1min"
ticker_folder_names="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
tickers="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
resample_rule="1min"
point_cloud_types="return tail_return return_ts"
lags="1 10 20 30 40 50 60 70"
python ../evaluate_simulations.py \
--initial_seed ${initial_seed} \
--significant_figures ${significant_figures} \
--config_path ${config_path} \
--specific_name ${specific_name} \
--txts_path ${txts_path} \
--num_simulations ${num_simulations} \
--use_simulator_given_runner \
--resample_mid \
--resample_rule ${resample_rule} \
--tick_dfs_path ${tick_dfs_path} \
--ohlcv_dfs_path ${ohlcv_dfs_path} \
--all_time_ohlcv_dfs_path ${all_time_ohlcv_dfs_path} \
--transactions_path ${transactions_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--figs_save_path ${figs_save_path} \
--results_save_path ${results_save_path} \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_file_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type} \
--lags ${lags}