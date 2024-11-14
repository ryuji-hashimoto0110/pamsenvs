import argparse
import json
from pams.runners import Runner
from pams.runners import SequentialRunner
import random
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from envs.agents import LeaderAwareLLMAgent
from envs.agents import LiquidityProviderAgent
from envs.events import DividendProvider
from envs.events import LeaderAwareMarketInitializer
from envs.events import LeadersPrioritizer
from envs.markets import LeaderAwareMarket
from logs import PortfolioSaver
from typing import Any
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--num_simulations", type=int, default=1)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--csvs_path", type=str)
    parser.add_argument("--record_ofi", action="store_true")
    parser.add_argument("--record_leader_board", action="store_true")
    parser.add_argument("--record_signal_description", action="store_true")
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    config_path: Path = pathlib.Path(all_args.config_path).resolve()
    config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    num_simulations: int = all_args.num_simulations
    all_csvs_path: Path = pathlib.Path(all_args.csvs_path).resolve()
    record_ofi: bool = all_args.record_ofi
    record_leader_board: bool = all_args.record_leader_board
    record_signal_description: bool = all_args.record_signal_description
    for i in range(num_simulations):
        print(f"{i=}, {record_ofi=}, {record_leader_board=}, {record_signal_description=}")
        csvs_path: Path = all_csvs_path / f"{i}"
        if not csvs_path.exists():
            csvs_path.mkdir(parents=True)   
        saver = PortfolioSaver(
            dfs_save_path=csvs_path,
            record_ofi=record_ofi,
            record_leader_board=record_leader_board,
            record_signal_description=record_signal_description
        )
        runner: Runner = SequentialRunner(
            settings=config,
            prng=random.Random(initial_seed+i),
            logger=saver
        )
        runner.class_register(LeaderAwareLLMAgent)
        runner.class_register(LiquidityProviderAgent)
        runner.class_register(LeaderAwareMarket)
        runner.class_register(DividendProvider)
        runner.class_register(LeadersPrioritizer)
        runner.class_register(LeaderAwareMarketInitializer)
        runner.main()

if __name__ == "__main__":
    main(sys.argv[1:])