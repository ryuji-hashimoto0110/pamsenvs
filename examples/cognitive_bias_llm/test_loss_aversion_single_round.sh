num_simulations=100
csv_path="results/loss_aversion/claude_single_round_uptrend_ath.csv"
temp=1.0
llm_name="claude-sonnet"
python test_loss_aversion_single_round.py \
--num_simulations ${num_simulations} \
--csv_path ${csv_path} \
--temp ${temp} \
--llm_name ${llm_name} \
--is_uptrend \
--is_peak