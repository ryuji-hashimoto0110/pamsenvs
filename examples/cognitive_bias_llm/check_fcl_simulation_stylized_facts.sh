initial_seed=42
significant_figures=10
config_path="fcl_config.json"
specific_name="fcl_rate001"
txts_path="../../datas/artificial_datas/flex_txt/fcl_simulation/fcl_rate001"
num_simulations=100
resample_rule="1min"
tick_dfs_path="../../datas/artificial_datas/flex_csv/fcl_simulation/fcl_rate001"
ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/fcl_simulation/fcl_rate001"
all_time_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/all_time/fcl_simulation"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_path="../../imgs/fcl_simulation/fcl_rate001"
results_save_path="../../stylized_facts/results/fcl_simulation/fcl_rate001.csv"

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
--results_save_path ${results_save_path}
