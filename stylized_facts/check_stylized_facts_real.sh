tick_folder_path="../datas/real_datas/flex_csv"
new_ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/1min/all"
transactions_save_folder_path="../datas/real_datas/intraday/flex_transactions/1min/all"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
resample_rule="1min"
specific_name=""
figs_folder="../imgs/1min/all"
session1_end_time_str="11:30:00.000000"
session2_start_time_str="12:30:00.000000"
results_folder="./results"
results_csv_name="1min_all.csv"
python check_stylized_facts.py \
--tick_folder_path ${tick_folder_path} \
--new_ohlcv_folder_path ${new_ohlcv_folder_path} \
--resample_rule ${resample_rule} \
--transactions_save_folder_path ${transactions_save_folder_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--specific_name ${specific_name} \
--is_real \
--choose_full_size_df \
--results_folder ${results_folder} \
--figs_folder ${figs_folder} \
--session1_end_time_str ${session1_end_time_str} \
--session2_start_time_str ${session2_start_time_str} \
--results_csv_name ${results_csv_name}