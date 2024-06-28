tickers="3407 4188 4568 5020 6502 9202"
daily_ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/1min"
all_time_ohlcv_folder_path="../datas/real_datas/intraday/flex_ohlcv/all_time"
start_year="2015"
start_month="1"
start_day="5"
end_year="2021"
end_month="8"
end_day="20"
python process_ohlcv.py \
--tickers ${tickers} \
--daily_ohlcv_folder_path ${daily_ohlcv_folder_path} \
--all_time_ohlcv_folder_path ${all_time_ohlcv_folder_path} \
--start_year ${start_year} \
--end_year ${end_year}
