rollout_length=1024
num_updates_per_rollout=10
batch_size=512
lr_actor=3e-04
lr_critic=3e-04
clip_eps=0.3
lmd=0.90
max_grad_norm=0.5
seed=50
actor_save_path="../../datas/checkpoints"
actor_best_save_name="pt_best"
actor_last_save_name="pt_last"
num_train_steps=60000000
eval_interval=105500
num_eval_episodes=5
agent_name="Agent"
config_path="config.json"
variable_ranges_path="variable_ranges.json"
obs_names="asset_ratio liquidable_asset_ratio inverted_buying_power "\
"log_return log_return_avg_cost volatility "\
"asset_volume_buy_orders_ratio asset_volume_sell_orders_ratio "\
"blurred_fundamental_return skill_boundedness discount_factor"
action_names="order_price_scale order_volume_scale"
depth_range=0.05
limit_order_range=0.05
max_order_volume=20
short_selling_penalty=0.1
cash_shortage_penalty=0.1
liquidity_penalty=0.003
liquidity_penalty_decay=1
initial_fundamental_penalty=0.03
fundamental_penalty_decay=1
agent_trait_memory=0.00
sigmas="0.008"
gammas="0.90"
device="cuda:0"
proceding_figures_path="imgs/last_layer_scale001"
python train_pt_inspired_rl.py \
--algo_name "ippo" \
--rollout_length $rollout_length \
--num_updates_per_rollout $num_updates_per_rollout \
--batch_size $batch_size \
--lr_actor $lr_actor \
--lr_critic $lr_critic \
--clip_eps $clip_eps \
--lmd $lmd \
--max_grad_norm $max_grad_norm \
--seed $seed \
--num_train_steps $num_train_steps \
--eval_interval $eval_interval \
--num_eval_episodes $num_eval_episodes \
--obs_names $obs_names \
--action_names $action_names \
--depth_range $depth_range \
--limit_order_range $limit_order_range \
--max_order_volume $max_order_volume \
--short_selling_penalty $short_selling_penalty \
--cash_shortage_penalty $cash_shortage_penalty \
--liquidity_penalty $liquidity_penalty \
--liquidity_penalty_decay $liquidity_penalty_decay \
--initial_fundamental_penalty $initial_fundamental_penalty \
--fundamental_penalty_decay $fundamental_penalty_decay \
--agent_trait_memory $agent_trait_memory \
--config_path $config_path \
--variable_ranges_path $variable_ranges_path \
--actor_save_path $actor_save_path \
--actor_best_save_name $actor_best_save_name \
--actor_last_save_name $actor_last_save_name \
--proceding_figures_path $proceding_figures_path \
--agent_name $agent_name \
--device $device \
--sigmas $sigmas \
--gammas $gammas