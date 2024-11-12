num_simulations=30
csv_path="results_loss_aversion/single_round_pt325_ath325.csv"
current_price=325
all_time_higth=325
all_time_low=200
temp=1.0
python test_loss_aversion_single_round.py \
--num_simulations ${num_simulations} \
--csv_path ${csv_path} \
--current_price ${current_price} \
--all_time_high ${all_time_high} \
--all_time_low ${all_time_low}
