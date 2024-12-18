config_path="config.json"
variable_ranges_path="variable_ranges.json"
actor_save_path="../../datas/checkpoints"
actor_best_save_name="best.pth"
actor_last_save_name="last.pth"
agent_name="Agent"
python train_hetero_rl.py \
--config_path $config_path \
--variable_ranges_path $variable_ranges_path \
--actor_save_path $actor_save_path \
--actor_best_save_name $actor_best_save_name \
--actor_last_save_name $actor_last_save_name \
--agent_name $agent_name