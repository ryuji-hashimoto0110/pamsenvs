csv_folder_path="../datas/real_datas/bybit_csv"
tickers="ADAUSD BITUSD BTCUSD DOTUSD EOSUSD ETHUSD LTCUSD LUNAUSD MANAUSD SOLUSD XRPUSD"
start_date="20191001"
end_date="20240820"
python process_bybit.py \
--csv_folder_path ${csv_folder_path} \
--tickers ${tickers} \
--start_date ${start_date} \
--end_date ${end_date}