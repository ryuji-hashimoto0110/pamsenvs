import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(root_path))
from ots.evaluate_distances_real import create_ddevaluaters
from ots import DDEvaluater
from ots import OTGridSearcher
from rich import print
from rich.tree import Tree
from typing import Any
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    # for DDEvaluater
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ohlcv_folder_path", type=str, default=None)
    parser.add_argument("--ticker_folder_names", type=str, nargs="*", default=None)
    parser.add_argument("--ticker_file_names", type=str, nargs="*", default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--resample_rule", type=str, default=None)
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--lags", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--point_cloud_type", type=str,
        choices=["return", "tail_return", "rv_returns", "return_ts"]
    )
    # for OTGridSearcher
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--base_config_path", type=str, default=None)
    parser.add_argument("--target_variables_config_path", type=str, default=None)
    parser.add_argument("--temp_txts_path", type=str, default=None)
    parser.add_argument("--temp_tick_dfs_path", type=str, default=None)
    parser.add_argument("--temp_ohlcv_dfs_path", type=str, default=None)    
    parser.add_argument("--temp_all_time_ohlcv_dfs_path", type=str, default=None)
    parser.add_argument("--path_to_calc_point_clouds", type=str, default=None)
    parser.add_argument("--num_simulations", type=int, default=1500)
    parser.add_argument("--use_simulator_given_runner", action="store_true")
    parser.add_argument("--is_mood_aware", action="store_true")
    parser.add_argument("--transactions_path", type=str, default=None)
    parser.add_argument("--session1_transactions_file_name", type=str, default=None)
    parser.add_argument("--session2_transactions_file_name", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--results_save_path", type=str, default=None)
    parser.add_argument("--show_process", action="store_true")
    return parser

def create_otsearcher(
    all_args,
    dd_evaluaters: list[DDEvaluater],
    show_args: bool = True
) -> OTGridSearcher:
    initial_seed: int = all_args.initial_seed
    show_process: bool = all_args.show_process
    use_simulator_given_runner: bool = all_args.use_simulator_given_runner
    is_mood_aware: bool = all_args.is_mood_aware
    num_simulations: int = all_args.num_simulations
    num_points: int = all_args.num_points
    if show_args:
        print(f"initial_seed: {initial_seed} show_process: {show_process}")
        print(f"num_points: {num_points} num_simulations: {num_simulations}")
        print(f"use_simulator_given_runner: {use_simulator_given_runner}")
        print(f"is_mood_aware: {is_mood_aware}")
    base_config_path: Optional[str] = all_args.base_config_path
    target_variables_config_path: Optional[str] = all_args.target_variables_config_path
    temp_txts_path: Optional[str] = all_args.temp_txts_path
    temp_tick_dfs_path: Optional[str] = all_args.temp_tick_dfs_path
    temp_ohlcv_dfs_path: Optional[str] = all_args.temp_ohlcv_dfs_path
    temp_all_time_ohlcv_dfs_path: Optional[str] = all_args.temp_all_time_ohlcv_dfs_path
    path_to_calc_point_clouds: Optional[str] = all_args.path_to_calc_point_clouds
    results_save_path: Optional[str] = all_args.results_save_path
    if show_args:
        print(f"base_config_path: {base_config_path}")
        print(f"target_variables_config_path: {target_variables_config_path}")
        print(f"temp_txts_path: {temp_txts_path}")
        print(f"temp_tick_dfs_path: {temp_tick_dfs_path}")
        print(f"temp_ohlcv_dfs_path: {temp_ohlcv_dfs_path}")
        print(f"temp_all_time_ohlcv_dfs_path: {temp_all_time_ohlcv_dfs_path}")
        print(f"path_to_calc_point_clouds: {path_to_calc_point_clouds}")
        print()
    transactions_path: Optional[str] = all_args.transactions_path
    session1_transactions_file_name: Optional[str] = all_args.session1_transactions_file_name
    session2_transactions_file_name: Optional[str] = all_args.session2_transactions_file_name
    if show_args:
        tree: Tree = Tree(str(transactions_path))
        tree.add(session1_transactions_file_name)
        tree.add(session2_transactions_file_name)
        print("transactions path:")
        print(tree)
        print()
    ot_searcher: OTGridSearcher = OTGridSearcher(
        initial_seed=initial_seed,
        dd_evaluaters=dd_evaluaters,
        base_config_path=base_config_path,
        target_variables_config_path=target_variables_config_path,
        use_simulator_given_runner=use_simulator_given_runner,
        is_mood_aware=is_mood_aware,
        num_simulations=num_simulations,
        temp_txts_path=temp_txts_path,
        temp_tick_dfs_path=temp_tick_dfs_path,
        temp_ohlcv_dfs_path=temp_ohlcv_dfs_path,
        temp_all_time_ohlcv_dfs_path=temp_all_time_ohlcv_dfs_path,
        path_to_calc_point_clouds=path_to_calc_point_clouds,
        transactions_path=transactions_path,
        session1_transactions_file_name=session1_transactions_file_name,
        session2_transactions_file_name=session2_transactions_file_name,
        num_points=num_points,
    )
    return ot_searcher

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    dd_evaluaters: DDEvaluater = create_ddevaluaters(all_args)
    show_process: bool = all_args.show_process
    ot_searcher: OTGridSearcher = create_otsearcher(all_args, dd_evaluaters, show_process)
    results_save_path: Optional[str] = all_args.results_save_path
    print(f"results_save_path: {results_save_path}")
    print("start searching.")
    ot_searcher.simulate_all_combs(result_save_path=results_save_path)

if __name__ == "__main__":
    main(sys.argv[1:])