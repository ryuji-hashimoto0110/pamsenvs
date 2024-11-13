initial_seed=42
num_simulations=30
config_path="loss_aversion_moderate_config.json"
csvs_path="results/loss_aversion/moderate"

python test_loss_aversion.py \
--initial_seed ${initial_seed} \
--num_simulations ${num_simulations} \
--config_path ${config_path} \
--csvs_path ${csvs_path}