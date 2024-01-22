encoder_type="Transformer"

train_data_type="real"
valid_data_type="real"
test_data_type="real"

train_olhcv_name="aapl"
valid_olhcv_name="aapl"
test_olhcv_name="aapl"

train_csv_names="AAPL2017.csv"
valid_csv_names="AAPL2018.csv"
test_csv_names="AAPL2019.csv"

train_obs_num=13
valid_obs_num=13
test_obs_num=13

train_mean_std_dic_name="aapl2017.json"
test_mean_std_dic_name="aapl2017.json"

criterion_type="MSE"
optimizer_type="AdamW"
learning_rate=0.0003
num_epochs=100
batch_size=512
num_workers=2
best_save_name="transformer_aapl2017_best.pth"
last_save_name="transformer_aapl2017_last.pth"
seed=42
python train_models.py \
--encoder_type ${encoder_type} \
\
--train_data_type ${train_data_type} \
--valid_data_type ${valid_data_type} \
--test_data_type ${test_data_type} \
\
--train_olhcv_name ${train_olhcv_name} \
--valid_olhcv_name ${valid_olhcv_name} \
--test_olhcv_name ${test_olhcv_name} \
\
--train_csv_names ${train_csv_names} \
--valid_csv_names ${valid_csv_names} \
--test_csv_names ${test_csv_names} \
\
--train_obs_num ${train_obs_num} \
--valid_obs_num ${valid_obs_num} \
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