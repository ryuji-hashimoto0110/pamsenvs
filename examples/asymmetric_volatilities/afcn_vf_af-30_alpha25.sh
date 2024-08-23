initial_seed=42
significant_figures=10
config_path="afcn_vf_af-30_alpha25.json"
specific_name="afcn_vf_af-30_alpha25"
txts_path="../../datas/artificial_datas/flex_txt/asymmetric_volatility/afcn_vf_af-30_alpha25"
num_simulations=1500
resample_rule="1min"
tick_dfs_path="../../datas/artificial_datas/flex_csv/asymmetric_volatility/afcn_vf_af-30_alpha25"
ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/asymmetric_volatility/afcn_vf_af-30_alpha25"
all_time_ohlcv_dfs_path="../../datas/artificial_datas/intraday/flex_ohlcv/all_time/asymmetric_volatility"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_save_path="../../imgs/asymmetric_volatility/afcn_vf_af-30_alpha25"
results_save_path="../../stylized_facts/results/asymmetric_volatility/afcn_vf_af-30_alpha25.csv"
check_asymmetry_path="../../stylized_facts/check_asymmetry.R"

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
--check_asymmetry_path ${check_asymmetry_path}