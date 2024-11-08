initial_seed=42
num_simulations=1
config_path="moderate_config.json"
csvs_path="results"

python test_loss_aversion.py \
--initial_seed ${initial_seed} \
--num_simulations ${num_simulations} \
--config_path ${config_path} \
--csvs_path ${csvs_path}