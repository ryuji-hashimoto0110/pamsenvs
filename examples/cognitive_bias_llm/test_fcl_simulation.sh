initial_seed=42
num_simulations=30
fcl_rate=0.1
config_path="fcl_config.json"
csvs_path="results/fcl_simulation/llama3"

python test_fcl_simulation.py \
--initial_seed ${initial_seed} \
--num_simulations ${num_simulations} \
--fcl_rate ${fcl_rate} \
--config_path ${config_path} \
--csvs_path ${csvs_path}