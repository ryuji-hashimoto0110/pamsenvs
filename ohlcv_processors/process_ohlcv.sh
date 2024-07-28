tickers="4568 6502 7203 8306 9202 9437"
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
--start_month ${start_month} \
--start_day ${start_day} \
--end_year ${end_year} \
--end_month ${end_month} \
--end_day ${end_day}
