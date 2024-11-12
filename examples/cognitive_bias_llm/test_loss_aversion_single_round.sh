num_simulations=30
csv_path="results/loss_aversion/single_round_pt325_ath325.csv"
current_price=325
all_time_high=325
all_time_low=200
temp=1.0
llm_name="gpt-4o"
python test_loss_aversion_single_round.py \
--num_simulations ${num_simulations} \
--csv_path ${csv_path} \
--current_price ${current_price} \
--all_time_high ${all_time_high} \
--all_time_low ${all_time_low} \
--temp ${temp} \
--llm_name ${llm_name}
