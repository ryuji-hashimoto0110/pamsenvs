tick_folder_path="../datas/real_datas/flex_csv"
new_ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/5min/9202"
transactions_save_folder_path="../datas/real_datas/intraday/flex_transactions/5min/9202"
resample_rule="5min"
specific_name="9202"
figs_folder="../imgs/5min/9202"
session1_end_time_str="11:30:00.000000"
session2_start_time_str="12:30:00.00000"
results_folder="./results"
results_csv_name="5min_9202.csv"
python check_stylized_facts.py \
--tick_folder_path ${tick_folder_path} \
--new_ohlcv_folder_path ${new_ohlcv_folder_path} \
--resample_rule ${resample_rule} \
--transactions_save_folder_path ${transactions_save_folder_path} \
--specific_name ${specific_name} \
--is_real \
--choose_full_size_df \
--results_folder ${results_folder} \
--figs_folder ${figs_folder} \
--session1_end_time_str ${session1_end_time_str} \
--session2_start_time_str ${session2_start_time_str} \
--results_csv_name ${results_csv_name}