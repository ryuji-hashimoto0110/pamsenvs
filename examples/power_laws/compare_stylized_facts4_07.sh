initial_seed=1042
configs_folder_path="."
config_names="4_07.json" 
txt_save_folder_paths="../../datas/artificial_datas/flex_txt/4_07"
num_simulations=4000
resample_rule="1min"
tick_dfs_folder_paths="../../datas/artificial_datas/flex_csv/4_07"
ohlcv_dfs_folder_paths="../../datas/artificial_datas/intraday/flex_ohlcv/1min/4_07"
transactions_folder_path="../../datas/real_datas/intraday/flex_transactions/1min/all"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_paths="../../imgs/compare_stylized_facts/4_07"
results_save_paths="../../stylized_facts/results/4_07.csv"
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