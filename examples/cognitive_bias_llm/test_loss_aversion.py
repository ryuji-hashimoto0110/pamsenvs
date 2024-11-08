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
from envs.agents import LiquidityProviderAgent
from envs.agents import HistoryAwareLLMAgent
from logs import PortfolioSaver
from typing import Any
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--csvs_path", type=str)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    config_path: Path = pathlib.Path(all_args.config_path).resolve()
    config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    csvs_path: Path = pathlib.Path(all_args.csvs_path).resolve()
    saver = PortfolioSaver(dfs_save_path=csvs_path)
    runner: Runner = SequentialRunner(
        settings=config,
        prng=random.Random(initial_seed),
        logger=saver
    )
    runner.class_register(LiquidityProviderAgent)
    runner.class_register(HistoryAwareLLMAgent)
    runner.main()

if __name__ == "__main__":
    main(sys.argv[1:])

