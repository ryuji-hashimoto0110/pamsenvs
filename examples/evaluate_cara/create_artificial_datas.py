import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from envs.agents import aFCNAgent
from logs import VolumePriceSaver
from pams.runners.sequential import SequentialRunner
import random
from typing import Optional
from typing import TypeVar

MarketID = TypeVar("MarketID")
parent_datas_path: Path = root_path / "datas" / "artificial_datas"

def create_artificial_olhcvs(
    config_path: Path,
    datas_path: Path,
    market_id: MarketID,
    start_index: int,
    index_interval: int,
    data_num: int,
    seeds: Optional[list[int]] = None
) -> None:
    if seeds is None:
        seeds: list[int] = [i+1 for i in range(data_num)]
    assert len(seeds) == data_num
    if not datas_path.exists():
        datas_path.mkdir(parents=True)
    for seed in seeds:
        saver = VolumePriceSaver()
        runner = SequentialRunner(
            settings=config_path,
            prng=random.Random(seed),
            logger=saver,
        )
        runner.class_register(aFCNAgent)
        runner.main()
        save_path: Path = datas_path / f"{seed}.csv"
        saver.save_olhcv(
            market_id, start_index, index_interval, save_path
        )

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str,
                        help="name of config file.")
    parser.add_argument("--datas_name", type=str,
                        help="folder name to store datas.")
    parser.add_argument("--market_id", type=int, default=0,
                        help="target market id to store data. If there is only 1 market setting, it is not needed to specify.")
    parser.add_argument("--start_index", type=int, default=100,
                        help="start index to store data. Usually, specify the time steps that first session starts that order execution is allowed.")
    parser.add_argument("--index_interval", type=int, default=100,
                        help="fixed time interval to tally OLHCV.")
    parser.add_argument("--data_num", type=int, default=100,
                        help="number of data.")
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    config_name: str = all_args.config_name
    config_path: Path = curr_path / config_name
    datas_name: str = all_args.datas_name
    datas_path: Path = parent_datas_path / datas_name
    market_id: MarketID = all_args.market_id
    start_index: int = all_args.start_index
    index_interval: int = all_args.index_interval
    data_num: int = all_args.data_num
    create_artificial_olhcvs(
        config_path, datas_path, market_id, start_index, index_interval, data_num
    )

if __name__ == "__main__":
    main(sys.argv[1:])