initial_seed=42
num_simulations=30
config_path="herd_behavior_config.json"
csvs_path="results/herd_behavior/llama3_base"

python test_herd_behavior.py \
--initial_seed ${initial_seed} \
--num_simulations ${num_simulations} \
--config_path ${config_path} \
--csvs_path ${csvs_path} \
--record_ofi \
--record_leader_board \
--record_signal_description