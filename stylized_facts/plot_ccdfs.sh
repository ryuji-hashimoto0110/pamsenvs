ohlcv_dfs_paths="../datas/real_datas/intraday/flex_ohlcv/1s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/10s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/30s/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/5min/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/15min/9202 "\
resample_rules="1s 10s 30s 1min 5min 15min"
labels="1s(tail=) 10s(tail=) 30s(tail=) 1min(tail=) 5min=()"
colors="black indianred forestgreen darkblue darkcyan brown"
fig_save_path="../imgs/9202_ccdf_compare.pdf"
python plot_ccdfs.py \
--ohlcv_dfs_paths ${ohlcv_dfs_paths} \
--resample_rules ${resample_rules} \
--labels ${labels} \
--colors ${colors} \
--fig_save_path ${fig_save_path}