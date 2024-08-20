tickers="EOSUSD"
daily_ohlcv_folder_path="../datas/real_datas/intraday/bybit_ohlcv/1min"
all_time_ohlcv_folder_path="../datas/real_datas/intraday/bybit_ohlcv/all_time"
start_year="2019"
start_month="10"
start_day="1"
end_year="2024"
end_month="8"
end_day="19"
python process_ohlcv.py \
--tickers ${tickers} \
--daily_ohlcv_folder_path ${daily_ohlcv_folder_path} \
--all_time_ohlcv_folder_path ${all_time_ohlcv_folder_path} \
--start_year ${start_year} \
--start_month ${start_month} \
--start_day ${start_day} \
--end_year ${end_year} \
--end_month ${end_month} \
--end_day ${end_day}