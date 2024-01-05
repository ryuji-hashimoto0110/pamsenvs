config_name="fcn_config.json"
datas_name="fcn"
market_id=0
start_index=100
index_interval=10
data_num=100
python create_artificial_datas.py --config_name ${config_name} --datas_name ${datas_name} --market_id ${market_id} \
--start_index ${start_index} --index_interval ${index_interval} --data_num ${data_num}
