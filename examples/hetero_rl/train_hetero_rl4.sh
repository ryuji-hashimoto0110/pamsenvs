rollout_length=128
num_updates_per_rollout=1
lr_actor=6e-05
lr_critic=8e-05
clip_eps=0.1
lmd=0.96
max_grad_norm=0.5
seed=42
actor_save_path="../../datas/checkpoints"
actor_best_save_name="best"
actor_last_save_name="last"
num_train_steps=4000000
eval_interval=21100
num_eval_episodes=3
agent_name="Agent"
config_path="config.json"
variable_ranges_path="variable_ranges.json"
obs_names="asset_ratio liquidable_asset_ratio inverted_buying_power "\
"log_return volatility asset_volume_buy_orders_ratio asset_volume_sell_orders_ratio "\
"blurred_fundamental_return skill_boundedness risk_aversion_term discount_factor"
action_names="order_price_scale order_volume_scale"
depth_range=0.03
limit_order_range=0.05
max_order_volume=50
short_selling_penalty=1.0
cash_shortage_penalty=1.0
execution_vonus=0.15
initial_fundamental_penalty=3.0
fundamental_penalty_decay=0.99
agent_trait_memory=0.9
sigmas="0.03 0.04"
alphas="0.00 0.30 0.60 0.90 2.00"
gammas="0.80 0.85 0.90 0.95 0.999"
device="cuda:1"
python train_hetero_rl.py \
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
--execution_vonus $execution_vonus \
--initial_fundamental_penalty $initial_fundamental_penalty \
--fundamental_penalty_decay $fundamental_penalty_decay \
--agent_trait_memory $agent_trait_memory \
--config_path $config_path \
--variable_ranges_path $variable_ranges_path \
--actor_save_path $actor_save_path \
--actor_best_save_name $actor_best_save_name \
--actor_last_save_name $actor_last_save_name \
--agent_name $agent_name \
--device $device \
--sigmas $sigmas \
--alphas $alphas \
--gammas $gammas