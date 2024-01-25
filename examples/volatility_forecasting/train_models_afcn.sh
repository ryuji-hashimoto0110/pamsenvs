encoder_type="Transformer"

train_data_type="artificial"
test_data_type="real"

train_olhcv_name="afcn/random"
test_olhcv_name="aapl"

test_csv_names="AAPL2019.csv"

train_obs_num=30
test_obs_num=13

train_mean_std_dic_name="aapl2017.json"
test_mean_std_dic_name="aapl2017.json"

criterion_type="MSE"
optimizer_type="AdamW"
learning_rate=0.00001
num_epochs=50
batch_size=512
num_workers=2
best_save_name="transformer_afcn_random_best.pth"
last_save_name="transformer_afcn_random_last.pth"
seed=42
python train_models.py \
--encoder_type ${encoder_type} \
\
--train_data_type ${train_data_type} \
--test_data_type ${test_data_type} \
\
--train_olhcv_name ${train_olhcv_name} \
--test_olhcv_name ${test_olhcv_name} \
\
--test_csv_names ${test_csv_names} \
\
--train_obs_num ${train_obs_num} \
--test_obs_num ${test_obs_num} \
\
--train_mean_std_dic_name ${train_mean_std_dic_name} \
--test_mean_std_dic_name ${test_mean_std_dic_name} \
\
--criterion_type ${criterion_type} \
--optimizer_type ${optimizer_type} \
--learning_rate ${learning_rate} \
--num_epochs ${num_epochs} \
--batch_size ${batch_size} \
--num_workers ${num_workers} \
--best_save_name ${best_save_name} \
--last_save_name ${last_save_name} \
--seed ${seed}