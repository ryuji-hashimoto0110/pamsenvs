ohlcv_dfs_paths="../datas/real_datas/intraday/flex_ohlcv/1s/all "\
"../datas/real_datas/intraday/flex_ohlcv/10s/all "\
"../datas/real_datas/intraday/flex_ohlcv/30s/all "\
"../datas/real_datas/intraday/flex_ohlcv/1min/all "\
"../datas/real_datas/intraday/flex_ohlcv/5min/all "\
"../datas/real_datas/intraday/flex_ohlcv/15min/all "\
resample_rules="1s 10s 30s 1min 5min 15min"
labels="1s(tail=) 10s(tail=) 30s(tail=) 1min(tail=) 5min(tail=) 15min(tail=)"
colors="black indianred forestgreen darkblue darkcyan brown"
fig_save_path="../imgs/ccdf_compare_all.pdf"
python plot_ccdfs.py \
--ohlcv_dfs_paths ${ohlcv_dfs_paths} \
--resample_rules ${resample_rules} \
--labels ${labels} \
--colors ${colors} \
--fig_save_path ${fig_save_path}