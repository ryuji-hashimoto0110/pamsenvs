import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(root_path))
from datetime import date
import numpy as np
from numpy import ndarray
from ots import DDEvaluater
from ots.evaluate_distances_real import create_ddevaluaters
from stylized_facts import SimulationEvaluater
from typing import Any
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    # SimulationEvaluater
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--significant_figures", type=int, default=10)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--specific_name", type=str, default=None)
    parser.add_argument("--txts_path", type=str, default=None)
    parser.add_argument("--num_simulations", type=int, default=2000)
    parser.add_argument("--use_simulator_given_runner", action="store_true")
    parser.add_argument("--resample_rule", type=str, default="1min")
    parser.add_argument("--resample_mid", action="store_true")
    parser.add_argument("--is_mood_aware", action="store_true")
    parser.add_argument("--is_wc_rate_aware", action="store_true")
    parser.add_argument("--tick_dfs_path", type=str, default=None)
    parser.add_argument("--ohlcv_dfs_path", type=str, default=None)
    parser.add_argument("--all_time_ohlcv_dfs_path", type=str, default=None)
    parser.add_argument("--transactions_path", type=str, default=None)
    parser.add_argument("--session1_transactions_file_name", type=str, default=None)
    parser.add_argument("--session2_transactions_file_name", type=str, default=None)
    parser.add_argument("--figs_save_path", type=str, default=None)
    parser.add_argument("--results_save_path", type=str, default=None)
    parser.add_argument("--check_asymmetry", action="store_true")
    parser.add_argument("--check_asymmetry_path", type=str, default="check_asymmetry.R")
    parser.add_argument("--check_ath_return", action="store_true")
    parser.add_argument("--check_ath_return_path", type=str, default="check_ath_return.R")
    # DDEvaluater
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ohlcv_folder_path", type=str, default=None)
    parser.add_argument("--ticker_folder_names", type=str, nargs="*", default=None)
    parser.add_argument("--ticker_file_names", type=str, nargs="*", default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--lags", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--point_cloud_type", type=str,
        choices=["return", "tail_return", "return_ts", "rv_returns"]
    )
    parser.add_argument("--n_samples", type=int, default=100)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    significant_figures: int = all_args.significant_figures
    config_path: Optional[str] = all_args.config_path
    specific_name: Optional[str] = all_args.specific_name
    txts_path: Optional[str] = all_args.txts_path
    resample_rule: Optional[str] = all_args.resample_rule
    resample_mid: bool = all_args.resample_mid
    is_mood_aware: bool = all_args.is_mood_aware
    is_wc_rate_aware: bool = all_args.is_wc_rate_aware
    num_simulations: int = all_args.num_simulations
    use_simulator_given_runner: bool = all_args.use_simulator_given_runner
    tick_dfs_path: Optional[str] = all_args.tick_dfs_path
    ohlcv_dfs_path: Optional[str] = all_args.ohlcv_dfs_path
    all_time_ohlcv_dfs_path: Optional[str] = all_args.all_time_ohlcv_dfs_path
    transactions_path: Optional[str] = all_args.transactions_path
    session1_transactions_file_name: Optional[str] = all_args.session1_transactions_file_name
    session2_transactions_file_name: Optional[str] = all_args.session2_transactions_file_name
    figs_save_path: Optional[str] = all_args.figs_save_path
    results_save_path: Optional[str] = all_args.results_save_path
    check_asymmetry: bool = all_args.check_asymmetry
    check_asymmetry_path: str = all_args.check_asymmetry_path
    check_ath_return: bool = all_args.check_ath_return
    check_ath_return_path: str = all_args.check_ath_return_path
    evaluater = SimulationEvaluater(
        initial_seed=initial_seed, significant_figures=significant_figures,
        config_path=config_path, specific_name=specific_name,
        txts_path=txts_path, resample_rule=resample_rule,
        resample_mid=resample_mid,
        is_mood_aware=is_mood_aware, is_wc_rate_aware=is_wc_rate_aware,
        tick_dfs_path=tick_dfs_path, ohlcv_dfs_path=ohlcv_dfs_path,
        all_time_ohlcv_dfs_path=all_time_ohlcv_dfs_path, transactions_path=transactions_path,
        session1_transactions_file_name=session1_transactions_file_name,
        session2_transactions_file_name=session2_transactions_file_name,
        figs_save_path=figs_save_path, results_save_path=results_save_path
    )
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    if (
        config_path is not None and txts_path is not None
    ):
        start_date, end_date = evaluater.simulate_multiple_times(
            num_simulations=num_simulations,
            use_simulator_given_runner=use_simulator_given_runner
        )
    if (
        txts_path is not None and
        tick_dfs_path is not None
    ):
        if 0 < len(list(evaluater.txts_path.iterdir())):
            evaluater.process_flex()
    if (
        tick_dfs_path is not None and
        ohlcv_dfs_path is not None and
        transactions_path is not None and
        session1_transactions_file_name is not None and
        session2_transactions_file_name is not None
    ):
        if (
            (
                check_asymmetry or
                check_ath_return
            ) and
            start_date is not None and
            end_date is not None
        ):
            evaluater.check_stylized_facts(
                check_asymmetry=check_asymmetry,
                check_asymmetry_path=check_asymmetry_path,
                check_ath_return=check_ath_return,
                check_ath_return_path=check_ath_return_path,
                start_date=start_date, end_date=end_date
            )
        else:
            evaluater.check_stylized_facts(check_asymmetry=False)
            if (
                start_date is not None and
                end_date is not None
            ):
                evaluater.concat_ohlcv(
                    start_date=start_date, end_date=end_date
                )
    if all_args.ohlcv_folder_path is not None:
        dd_evaluater: DDEvaluater = create_ddevaluaters(all_args)[0]
        n_samples: int = all_args.n_samples
        real_tickers: list[str | int] = list(dd_evaluater.ticker_path_dic.keys())
        start_date_str: str = start_date.strftime(format='%Y%m%d')
        end_date_str: str = end_date.strftime(format='%Y%m%d')
        all_time_ohlcv_df_path: Path = pathlib.Path(all_time_ohlcv_dfs_path).resolve() / \
            f"{specific_name}_{start_date_str}_{end_date_str}.csv"
        dd_evaluater.add_ticker_path(
            ticker=specific_name, path=all_time_ohlcv_df_path
        )
        art_point_cloud: ndarray = dd_evaluater.get_point_cloud_from_ticker(
            ticker=specific_name, num_points=n_samples, save2dic=False, return_statistics=False
        )
        ot_distance_dic: dict[int | str, float] = {}
        for ticker in real_tickers:
            real_point_cloud: ndarray = dd_evaluater.get_point_cloud_from_ticker(
                ticker=ticker, num_points=n_samples, save2dic=False, return_statistics=False
            )
            ot_distance: float = dd_evaluater.calc_ot_distance(
                art_point_cloud, real_point_cloud, is_per_bit=True
            )
            ot_distance_dic[ticker] = ot_distance
        print(f"OT distances:")
        print(ot_distance_dic)
        ot_distances: list[float] = list(ot_distance_dic.values())
        print(f"ot(median):{np.median(ot_distances):.5f} ot(average):{np.mean(ot_distances):.5f} ot(std):{np.std(ot_distances):.5f}")
        
if __name__ == "__main__":
    main(sys.argv[1:])
