ohlcv_dfs_paths="../datas/real_datas/intraday/flex_ohlcv/1s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/10s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/30s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/5min/9202"
resample_rules="1s 10s 30s 1min 5min"
labels="1s(tail=3.743) 10s(tail=3.227) 30s(tail=3.062) 1min(tail=2.600) 5min=(2.430)"
colors="black indianred forestgreen darkblue darkcyan"
fig_save_path="../imgs/9202_ccdf_compare.pdf"
python plot_ccdfs.py \
--ohlcv_dfs_paths ${ohlcv_dfs_paths} \
--resample_rules ${resample_rules} \
--labels ${labels} \
--colors ${colors} \
--fig_save_path ${fig_save_path}