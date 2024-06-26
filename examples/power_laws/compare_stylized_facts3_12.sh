initial_seed=42
configs_folder_path="."
config_names="3_12.json" 
txt_save_folder_paths="../../datas/artificial_datas/flex_txt/3_12"
num_simulations=1000
resample_rule="1min"
tick_dfs_folder_paths="../../datas/artificial_datas/flex_csv/3_12"
ohlcv_dfs_folder_paths="../../datas/artificial_datas/intraday/flex_ohlcv/1min/3_12"
transactions_folder_path="../../datas/real_datas/intraday/flex_transactions/1min/all"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_paths="../../imgs/compare_stylized_facts/3_12"
results_save_paths="../../stylized_facts/results/3_12.csv"
python compare_stylized_facts.py \
--initial_seed ${initial_seed} \
--configs_folder_path ${configs_folder_path} \
--config_names ${config_names} \
--resample_rule ${resample_rule} \
--txt_save_folder_paths ${txt_save_folder_paths} \
--num_simulations ${num_simulations} \
--tick_dfs_folder_paths ${tick_dfs_folder_paths} \
--ohlcv_dfs_folder_paths ${ohlcv_dfs_folder_paths} \
--transactions_folder_path ${transactions_folder_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--figs_save_paths ${figs_save_paths} \
--results_save_paths ${results_save_paths}