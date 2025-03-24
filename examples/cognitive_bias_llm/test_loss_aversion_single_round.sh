num_simulations=100
csv_path="results/loss_aversion/llama3_single_round_uptrend_ath.csv"
temp=1.0
llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
device="cuda:0"
python test_loss_aversion_single_round.py \
--num_simulations ${num_simulations} \
--csv_path ${csv_path} \
--temp ${temp} \
--llm_name ${llm_name} \
--device ${device} \
--is_uptrend \
--is_peak