initial_seed=42
significant_figures=10
config_path="world_fcn_config.json"
specific_name="world_fcn"
txts_path="../../datas/artificial_datas/flex_txt/world_fcn"
num_simulations=10
resample_rule="1min"
tick_dfs_path="../../datas/artificial_datas/flex_csv/asymmetric_volatility/world_fcn"
ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/asymmetric_volatility/world_fcn"
all_time_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/all_time/asymmetric_volatility/world_fcn"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_path="../../imgs/asymmetric_volatility/world_fcn"
results_save_path="../../stylized_facts/results/asymmetric_volatility/world_fcn.csv"
seed=42
ohlcv_folder_path="../../datas/real_datas/intraday/flex_ohlcv/all_time"

python ../evaluate_simulations.py \
--initial_seed ${initial_seed} \
--significant_figures ${significant_figures} \
--config_path ${config_path} \
--specific_name ${specific_name} \
--txts_path ${txts_path} \
--num_simulations ${num_simulations} \
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
--resample_rule ${resample_rule}
