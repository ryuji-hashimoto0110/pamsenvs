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
from envs.agents import HistoryAwareLLMAgent
from envs.agents import LiquidityProviderAgent
from envs.markets import TotalTimeAwareMarket
from logs import PortfolioSaver
from typing import Any
from typing import Optional
import warnings
warnings.simplefilter("ignore")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--num_simulations", type=int, default=1)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--csvs_path", type=str)
    parser.add_argument("--market_name", type=str, default="Market")
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    config_path: Path = pathlib.Path(all_args.config_path).resolve()
    initial_config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    num_simulations: int = all_args.num_simulations
    market_name: str = all_args.market_name
    all_csvs_path: Path = pathlib.Path(all_args.csvs_path).resolve()
    for i in range(num_simulations):
        prng = random.Random(initial_seed+i)
        csvs_path: Path = all_csvs_path / f"{i}"
        if not csvs_path.exists():
            csvs_path.mkdir(parents=True)
        saver = PortfolioSaver(dfs_save_path=csvs_path)
        config = initial_config.copy()
        initial_drift: float = initial_config[market_name]["fundamentalDrift"]
        drift: float = prng.uniform(a=initial_drift-0.003, b=initial_drift+0.003)
        initial_vola: float = initial_config[market_name]["fundamentalVolatility"]
        vola: float = prng.uniform(a=0.005, b=initial_vola)
        config[market_name]["fundamentalDrift"] = drift
        config[market_name]["fundamentalVolatility"] = vola
        runner: Runner = SequentialRunner(
            settings=config,
            prng=random.Random(initial_seed+i),
            logger=saver
        )
        runner.class_register(LiquidityProviderAgent)
        runner.class_register(HistoryAwareLLMAgent)
        runner.class_register(TotalTimeAwareMarket)
        runner.main()

if __name__ == "__main__":
    main(sys.argv[1:])

