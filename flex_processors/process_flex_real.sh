txt_folder_path="../datas/real_datas/flex_txt"
csv_folder_path="../datas/real_datas/flex_csv"
flex_downloader_path="~/flex_full_processed_downloader/downloader.rb"
tickers="1605 4502 6273 6981 8001 8601 9432 1878 4503 6301 6988 8002 8604 9433 1925 4523 6326 7011 8031 8630 9437 1928 6367 7201 8035 8725 9502 1963 4578 6501 7202 8053 8750 9503 2502 4661 7203 8058 8766 9531 2503 4755 6503 7261 8113 8795  9532 2802 4901 6594 7267 8267 8801 9735 2914 4911 6702 7269 8306 8802 9983 3382 5020 6752 7270 8308 8830 9984 3402 5108 6758 7741 8309 9020 5401 6861 7751 8316 9021 4063 5411 6902 7752 8332 9022 5713 6954 7912 8411 9064 4452 5802 6971 7974 8591"
start_date="20150708"
end_date="20170101"
quote_num=10
python process_flex.py \
--txt_folder_path ${txt_folder_path} \
--csv_folder_path ${csv_folder_path} \
--flex_downloader_path ${flex_downloader_path} \
--tickers ${tickers} \
--start_date ${start_date} \
--end_date ${end_date} \
--quote_num ${quote_num}