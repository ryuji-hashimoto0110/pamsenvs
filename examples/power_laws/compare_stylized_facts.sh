initial_seed=42
configs_folder_path="."
config_names="0.json 1.json 2.json 3.json 4.json 5.json 6.json 7.json" 
txt_save_folder_paths="../../datas/artificial_datas/flex_txt/0 "\
"../../datas/artificial_datas/flex_txt/1 "\
"../../datas/artificial_datas/flex_txt/2 "\
"../../datas/artificial_datas/flex_txt/3 "\
"../../datas/artificial_datas/flex_txt/4 "\
"../../datas/artificial_datas/flex_txt/5 "\
"../../datas/artificial_datas/flex_txt/6 "\
"../../datas/artificial_datas/flex_txt/7"
num_simulations=2
resample_rule="1min"
tick_dfs_folder_paths="../../datas/artificial_datas/flex_csv/0 "\
"../../datas/artificial_datas/flex_csv/1 "\
"../../datas/artificial_datas/flex_csv/2 "\
"../../datas/artificial_datas/flex_csv/3 "\
"../../datas/artificial_datas/flex_csv/4 "\
"../../datas/artificial_datas/flex_csv/5 "\
"../../datas/artificial_datas/flex_csv/6 "\
"../../datas/artificial_datas/flex_csv/7"
ohlcv_dfs_folder_paths="../../datas/artificial_datas/intraday/flex_ohlcv/1min/0 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/1 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/2 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/3 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/4 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/5 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/6 "\
"../../datas/artificial_datas/intraday/flex_ohlcv/1min/7"
transactions_folder_path="../../datas/real_datas/intraday/flex_transactions/1min/9202"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_paths="../../imgs/compare_stylized_facts/0 "\
"../../imgs/compare_stylized_facts/1 "\
"../../imgs/compare_stylized_facts/2 "\
"../../imgs/compare_stylized_facts/3 "\
"../../imgs/compare_stylized_facts/4 "\
"../../imgs/compare_stylized_facts/5 "\
"../../imgs/compare_stylized_facts/6 "\
"../../imgs/compare_stylized_facts/7"
results_save_paths="../../stylized_facts/results/0.csv "\
"../../stylized_facts/results/1.csv "\
"../../stylized_facts/results/2.csv "\
"../../stylized_facts/results/3.csv "\
"../../stylized_facts/results/4.csv "\
"../../stylized_facts/results/5.csv "\
"../../stylized_facts/results/6.csv "\
"../../stylized_facts/results/7.csv"
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