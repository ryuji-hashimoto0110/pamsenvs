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
from envs.agents import HistoryAwareFCLAgent
from logs import PortfolioSaver
from typing import Any
from typing import Optional
import warnings
warnings.simplefilter("ignore")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--num_simulations", type=int, default=1)
    parser.add_argument("--fcl_rate", type=float, default=0.1)
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
    fcl_rate: float = all_args.fcl_rate
    num_agents: int = initial_config["FCNAgents"]["numAgents"]
    initial_config["HistoryAwareFCLAgents"]["numAgents"] = int(fcl_rate*num_agents)
    initial_config["FCNAgents"]["numAgents"] = \
        num_agents - initial_config["HistoryAwareFCLAgents"]["numAgents"]
    num_simulations: int = all_args.num_simulations
    all_csvs_path: Path = pathlib.Path(all_args.csvs_path).resolve()
    market_name: str = all_args.market_name
    for i in range(num_simulations):
        prng = random.Random(initial_seed+i)
        csvs_path: Path = all_csvs_path / f"{i}"
        if not csvs_path.exists():
            csvs_path.mkdir(parents=True)
        config = initial_config.copy()
        initial_drift: float = initial_config[market_name]["fundamentalDrift"]
        vol: float = initial_config[market_name]["fundamentalVolatility"]
        drift: float = prng.uniform(a=initial_drift+0.5*vol**2-0.002, b=initial_drift+0.5*vol**2+0.002)
        config[market_name]["fundamentalDrift"] = drift
        print(f"{i=}")
        print(config[market_name])
        saver = PortfolioSaver(dfs_save_path=csvs_path)
        runner: Runner = SequentialRunner(
            settings=config,
            prng=random.Random(initial_seed+i),
            logger=saver
        )
        runner.class_register(HistoryAwareFCLAgent)
        runner.main()

if __name__ == "__main__":
    main(sys.argv[1:])