initial_seed=42
config_path="moderate_config.json"
csvs_path="results"

python test_loss_aversion.py \
--initial_seed ${initial_seed} \
--config_path ${config_path} \
--csvs_path ${csvs_path}