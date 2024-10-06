ohlcv_dfs_paths="../datas/real_datas/intraday/flex_ohlcv/1min/2802 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/3382 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/4063 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/4452 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/4568 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/4578 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/6501 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/6502 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/7203 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/7267 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8001 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8035 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8058 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8306 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/8411 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9202 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9613 "\
"../datas/real_datas/intraday/flex_ohlcv/1min/9984"
resample_rules="1min 1min 1min 1min 1min 1min 1min 1min 1min 1min "\
"1min 1min 1min 1min 1min 1min 1min 1min"
labels="2802 3382 4063 4452 4568 4578 6501 6502 7203 7267 "\
"8001 8035 8058 8306 8411 9202 9613 9984"
colors="black indianred forestgreen darkblue darkcyan brown darkorange darkviolet slategray crimson "\
"aquamarine lime green yellow orange blue violet olive"
fig_save_path="../imgs/acorrs_real_all.pdf"
python plot_acorrs.py \
--ohlcv_dfs_paths ${ohlcv_dfs_paths} \
--resample_rules ${resample_rules} \
--labels ${labels} \
--colors ${colors} \
--fig_save_path ${fig_save_path}