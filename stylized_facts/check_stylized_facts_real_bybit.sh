tick_folder_path="../datas/real_datas/bybit_csv"
new_ohlcv_folder_path="../datas/real_datas/intraday/bybit_ohlcv/1min/EOSUSD"
transactions_save_folder_path="../datas/real_datas/intraday/bybit_transactions/1min/EOSUSD"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
resample_rule="1min"
specific_name="EOSUSD"
figs_folder="../imgs/1min/EOSUSD"
session1_end_time_str="23:59:59.999999"
results_folder="./results"
results_csv_name="1min_EOSUSD.csv"
python check_stylized_facts.py \
--tick_folder_path ${tick_folder_path} \
--new_ohlcv_folder_path ${new_ohlcv_folder_path} \
--resample_rule ${resample_rule} \
--transactions_save_folder_path ${transactions_save_folder_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--specific_name ${specific_name} \
--is_real \
--is_bybit \
--choose_full_size_df \
--results_folder ${results_folder} \
--figs_folder ${figs_folder} \
--session1_end_time_str ${session1_end_time_str} \
--results_csv_name ${results_csv_name}