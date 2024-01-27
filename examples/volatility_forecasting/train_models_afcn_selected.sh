encoder_type="Transformer"

train_data_type="artificial"
test_data_type="real"

train_olhcv_name="afcn/random"
test_olhcv_name="aapl"

train_csv_names="0.csv 10.csv 11.csv 12.csv 16.csv 162.csv 19.csv 25.csv 282.csv 32.csv "\
"379.csv 38.csv 385.csv 390.csv 391.csv 412.csv 460.csv 462.csv 498.csv 50.csv "\
"52.csv 590.csv 608.csv 61.csv 62.csv 64.csv 643.csv 65.csv 651.csv 69.csv 704.csv "\
"707.csv 741.csv 747.csv 755.csv 766.csv 77.csv 84.csv 85.csv 852.csv 853.csv 86.csv "\
"89.csv 91.csv 918.csv 92.csv 974.csv 975.csv 987.csv 998.csv"
test_csv_names="AAPL2019.csv"

train_obs_num=30
test_obs_num=13

train_mean_std_dic_name="afcn_random_selected.json"
test_mean_std_dic_name="aapl2017.json"

criterion_type="MSE"
optimizer_type="AdamW"
learning_rate=0.0001
num_epochs=60
batch_size=512
num_workers=2
best_save_name="transformer_afcn_random_selected_best.pth"
last_save_name="transformer_afcn_random_selectef_last.pth"
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
--train_csv_names ${train_csv_names} \
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