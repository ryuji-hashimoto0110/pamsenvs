ohlcv_folder_path="../../datas/real_datas/intraday/flex_ohlcv/1min"
ticker_folder_names="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
tickers="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 8001 8035 8058 8306 8411 9202 9613 9984"
resample_rule="1min"
point_cloud_types="return tail_return return_ts"
lags="1 10 20 30 40 50 60 70"
algo_name="ippo"
agent_name="Agent"
config_path="config.json"
variable_ranges_path="variable_ranges.json"
obs_names="asset_ratio liquidable_asset_ratio inverted_buying_power "\
"log_return volatility "\
"asset_volume_buy_orders_ratio asset_volume_sell_orders_ratio "\
"blurred_fundamental_return skill_boundedness risk_aversion_term discount_factor"
action_names="order_price_scale order_volume_scale"
depth_range=0.05
limit_order_range=0.05
max_order_volume=50
short_selling_penalty=1.0
cash_shortage_penalty=1.0
liquidity_penalty=0.1
liquidity_penalty_decay=1
initial_fundamental_penalty=3.0
fundamental_penalty_decay=1
agent_trait_memory=0.0
actor_folder_path="../../datas/checkpoints"
sigmas="0.002"
alphas="0.15"
gammas="0.80"
txts_save_path="../../datas/artificial_datas/flex_txt/hetero_rl/temp"
tick_dfs_save_path="../../datas/artificial_datas/flex_csv/hetero_rl/temp"
ohlcv_dfs_save_path="../../datas/artificial_datas/intraday/flex_ohlcv/1min/hetero_rl/temp"
transactions_path="../../datas/real_datas/intraday/flex_transactions/1min/7203"
session1_transactions_file_name="cumsum_scaled_transactions_session1.csv"
session2_transactions_file_name="cumsum_scaled_transactions_session2.csv"
figs_folder_path="../../imgs/hetero_rl"
market_name="Market"
decision_histories_save_path="../../datas/artificial_datas/hetero_rl_decision_histories/temp"
stylized_facts_folder_path="../../stylized_facts/results/hetero_rl"
ot_distances_save_path="ots.csv"
device="cuda:1"
python evaluate_hetero_rl.py \
--ohlcv_folder_path ${ohlcv_folder_path} \
--ticker_folder_names ${ticker_folder_names} \
--tickers ${tickers} \
--resample_rule ${resample_rule} \
--point_cloud_types ${point_cloud_types} \
--lags ${lags} \
--algo_name ${algo_name} \
--agent_name ${agent_name} \
--config_path ${config_path} \
--variable_ranges_path ${variable_ranges_path} \
--obs_names ${obs_names} \
--action_names ${action_names} \
--depth_range ${depth_range} \
--limit_order_range ${limit_order_range} \
--max_order_volume ${max_order_volume} \
--short_selling_penalty ${short_selling_penalty} \
--cash_shortage_penalty ${cash_shortage_penalty} \
--liquidity_penalty $liquidity_penalty \
--liquidity_penalty_decay $liquidity_penalty_decay \
--initial_fundamental_penalty $initial_fundamental_penalty \
--fundamental_penalty_decay $fundamental_penalty_decay \
--agent_trait_memory ${agent_trait_memory} \
--actor_folder_path ${actor_folder_path} \
--sigmas ${sigmas} \
--alphas ${alphas} \
--gammas ${gammas} \
--txts_save_path ${txts_save_path} \
--tick_dfs_save_path ${tick_dfs_save_path} \
--ohlcv_dfs_save_path ${ohlcv_dfs_save_path} \
--transactions_path ${transactions_path} \
--session1_transactions_file_name ${session1_transactions_file_name} \
--session2_transactions_file_name ${session2_transactions_file_name} \
--figs_folder_path ${figs_folder_path} \
--market_name ${market_name} \
--decision_histories_save_path ${decision_histories_save_path} \
--stylized_facts_folder_path ${stylized_facts_folder_path} \
--ot_distances_save_path ${ot_distances_save_path} \
--device ${device}

