initial_seed=42
significant_figures=10
config_path="afcn_vf_af30_alpha30.json"
specific_name="afcn_vf_af30_alpha30"
txts_path="../../datas/artificial_datas/flex_txt/asymmetric_volatility/afcn_vf_af30_alpha30"
num_simulations=1500
resample_rule="1min"
tick_dfs_path="../../datas/artificial_datas/flex_csv/asymmetric_volatility/afcn_vf_af30_alpha30"
ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/asymmetric_volatility/afcn_vf_af30_alpha30"
all_time_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/all_time/asymmetric_volatility"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_path="../../imgs/asymmetric_volatility/afcn_vf_af30_alpha30"
results_save_path="../../stylized_facts/results/asymmetric_volatility/afcn_vf_af30_alpha30.csv"
check_asymmetry_path="../../stylized_facts/check_asymmetry.R"
ohlcv_folder_path="../../datas/real_datas/intraday/flex_ohlcv/all_time"
ticker_file_names="2802_20150105_20210820.csv "\
"3382_20150105_20210820.csv "\
"4063_20150105_20210820.csv "\
"4452_20150105_20210820.csv "\
"4568_20150105_20210820.csv "\
"4578_20150105_20210820.csv "\
"6501_20150105_20210820.csv "\
"6502_20150105_20210820.csv "\
"7203_20150105_20210820.csv "\
"7267_20150105_20210820.csv "\
"8001_20150105_20210820.csv "\
"8035_20150105_20210820.csv "\
"8058_20150105_20210820.csv "\
"8306_20150105_20210820.csv "\
"8411_20150105_20210820.csv "\
"9202_20150105_20210820.csv "\
"9613_20150105_20210820.csv "\
"9984_20150105_20210820.csv"
tickers="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
point_cloud_type="rv_returns"

python ../evaluate_simulations.py \
--initial_seed ${initial_seed} \
--significant_figures ${significant_figures} \
--config_path ${config_path} \
--specific_name ${specific_name} \
--txts_path ${txts_path} \
--num_simulations ${num_simulations} \
--use_simulator_given_runner \
--resample_mid \
--resample_rule ${resample_rule} \
--tick_dfs_path ${tick_dfs_path} \
--ohlcv_dfs_path ${ohlcv_dfs_path} \
--all_time_ohlcv_dfs_path ${all_time_ohlcv_dfs_path} \
--transactions_path ${transactions_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--figs_save_path ${figs_save_path} \
--results_save_path ${results_save_path} \
--check_asymmetry \
--check_asymmetry_path ${check_asymmetry_path} \
--seed ${seed} \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_file_names ${ticker_file_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_type ${point_cloud_type}