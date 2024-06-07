txt_folder_path="../datas/real_datas/flex_txt"
csv_folder_path="../datas/real_datas/flex_csv"
flex_downloader_path="~/flex_full_processed_downloader/downloader.rb"
tickers="3407 4188 4568 5020 6502"
start_date="20150101"
end_date="20211231"
quote_num=10
python process_flex.py \
--txt_folder_path ${txt_folder_path} \
--csv_folder_path ${csv_folder_path} \
--flex_downloader_path ${flex_downloader_path} \
--tickers ${tickers} \
--start_date ${start_date} \
--end_date ${end_date} \
--quote_num ${quote_num}