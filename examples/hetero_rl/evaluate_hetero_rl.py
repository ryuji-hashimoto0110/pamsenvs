import argparse
from argparse import ArgumentParser
import json
from train_hetero_rl import convert_str2path
from train_hetero_rl import create_env
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from drl_algos import Evaluater
from drl_algos import IPPO
from ots.evaluate_distances_real import create_ddevaluaters
import torch
from typing import Any
from typing import Optional

def get_config() -> ArgumentParser:
    parser: ArgumentParser = argparse.ArgumentParser()
    # for DDEvaluater
    parser.add_argument("--ohlcv_folder_path", type=str, default=None)
    parser.add_argument("--ticker_folder_names", type=str, nargs="*", default=None)
    parser.add_argument("--ticker_file_names", type=str, nargs="*", default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--resample_rule", type=str, default=None)
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--lags", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--point_cloud_types", type=str, nargs="+",
        choices=["return", "tail_return", "rv_returns", "return_ts"]
    )
    # for algo, env
    parser.add_argument(
        "--algo_name", type=str, default="ippo", choices=["ippo"]
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--agent_name", type=str, default="heteroRLAgent")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--variable_ranges_path", type=str, required=False)
    parser.add_argument("--obs_names", type=str, nargs="+")
    parser.add_argument("--action_names", type=str, nargs="+")
    parser.add_argument("--depth_range", type=float, default=0.01)
    parser.add_argument("--limit_order_range", type=float, default=0.05)
    parser.add_argument("--max_order_volume", type=int, default=50)
    parser.add_argument("--short_selling_penalty", type=float, default=0.5)
    parser.add_argument("--execution_vonus", type=float, default=0.1)
    parser.add_argument("--agent_trait_memory", type=float, default=0.9)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    # for Evaluater
    parser.add_argument("--actor_load_path", type=str, default=None)
    parser.add_argument("--txts_save_path", type=str, default=None)
    parser.add_argument("--tick_dfs_save_path", type=str, default=None)
    parser.add_argument("--ohlcv_dfs_save_path", type=str, default=None)
    parser.add_argument("--transactions_path", type=str, default=None)
    parser.add_argument("--session1_transactions_file_name", type=str, default=None)
    parser.add_argument("--session2_transactions_file_name", type=str, default=None)
    parser.add_argument("--market_name", type=str, default=None)
    parser.add_argument("--decision_histories_save_path", type=str, default=None)
    parser.add_argument("--figs_save_path", type=str, default=None)
    parser.add_argument("--indicators_save_path", type=str, default=None)
    return parser

def main(args) -> None:
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    dd_evaluaters = create_ddevaluaters(all_args, multiple_ts_only=True)
    config_path: Path = convert_str2path(all_args.config_path, mkdir=False)
    config_dic: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    env, num_agents = create_env(all_args)
    ippo: IPPO = IPPO(
        device=all_args.device,
        obs_shape=(12,), action_shape=(2,), num_agents=num_agents,
        seed=all_args.seed,
    )
    evaluater: Evaluater = Evaluater(
        dd_evaluaters=dd_evaluaters, num_points=1500,
        env=env, algo=ippo, actor_load_path=convert_str2path(all_args.actor_load_path, False),
        txts_save_path=convert_str2path(all_args.txts_save_path, True),
        tick_dfs_save_path=convert_str2path(all_args.tick_dfs_save_path, True),
        ohlcv_dfs_save_path=convert_str2path(all_args.ohlcv_dfs_save_path, True),
        transactions_path=convert_str2path(all_args.transactions_path, False),
        session1_transactions_file_name=all_args.session1_transactions_file_name,
        session2_transactions_file_name=all_args.session2_transactions_file_name,
        market_name=all_args.market_name,
        decision_histories_save_path=convert_str2path(all_args.decision_histories_save_path, True),
        figs_save_path=convert_str2path(all_args.figs_save_path, True),
        indicators_save_path=convert_str2path(all_args.indicators_save_path, True)
    )
    evaluater.save_multiple_episodes()

if __name__ == "__main__":
    main(sys.argv[1:])