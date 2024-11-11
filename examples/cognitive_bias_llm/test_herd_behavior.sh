initial_seed=42
num_simulations=1
config_path="herd_behavior_of_lb_config.json"
csvs_path="results_herd_behavior_of_lb"

python test_herd_behavior.py \
--initial_seed ${initial_seed} \
--num_simulations ${num_simulations} \
--config_path ${config_path} \
--csvs_path ${csvs_path} \
--record_ofi \
--record_leader_board \
--record_signal_description