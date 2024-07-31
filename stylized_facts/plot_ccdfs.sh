ohlcv_dfs_paths="../datas/real_datas/intraday/flex_ohlcv/1min/3382 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/4568 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/6501 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/6502 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/7203 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8306 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8411 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9202"
resample_rules="1min 1min 1min 1min 1min 1min 1min 1min"
labels="3382 4568 6501 6502 7203 8306 8411 9202"
colors="black indianred forestgreen darkblue darkcyan brown darkorange darkviolet"
fig_save_path="../imgs/ccdf_real_all.pdf"
python plot_ccdfs.py \
--ohlcv_dfs_paths ${ohlcv_dfs_paths} \
--resample_rules ${resample_rules} \
--labels ${labels} \
--colors ${colors} \
--fig_save_path ${fig_save_path}