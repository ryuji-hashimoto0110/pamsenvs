ohlcv_folder_path="../datas/real_datas/flex_csv"
new_ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv"
specific_name="9202"
results_folder="./results"
figs_folder="imgs/9202"
results_csv_name="9202.csv"
python check_stylized_facts.py \
--ohlcv_folder_path ${ohlcv_folder_path} \
--new_ohlcv_folder_path ${new_ohlcv_folder_path} \
--specific_name ${specific_name} \
--choose_full_size_df \
--need_resample \
--results_folder ${results_folder} \
--figs_folder ${figs_folder} \
--results_csv_name ${results_csv_name}