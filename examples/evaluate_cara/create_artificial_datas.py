import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from envs.agents import aFCNAgent
from logs import DataMaker
from pams.runners.base import Runner
from typing import TypeVar

MarketID = TypeVar("MarketID")
parent_daily_datas_path: Path = root_path / "datas" / "artificial_datas" / "daily"
parent_intraday_datas_path: Path = root_path / "datas" / "artificial_datas" / "intraday"


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
    parser.add_argument("--daily_index_interval", type=int, default=100,
                        help="fixed time interval to tally daily OLHCV.")
    parser.add_argument("--intraday_index_interval", type=int, default=100,
                        help="fixed time interval to tally intraday OLHCV.")
    parser.add_argument("--data_num", type=int, default=100,
                        help="number of data.")
    return parser

class FCNwCARADataMaker(DataMaker):
    def class_register(self, runner: Runner):
        runner.class_register(aFCNAgent)

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    config_name: str = all_args.config_name
    config_path: Path = curr_path / config_name
    datas_name: str = all_args.datas_name
    daily_datas_path: Path = parent_daily_datas_path / datas_name
    intraday_datas_path: Path = parent_intraday_datas_path / datas_name
    market_id: MarketID = all_args.market_id
    start_index: int = all_args.start_index
    daily_index_interval: int = all_args.daily_index_interval
    intraday_index_interval: int = all_args.intraday_index_interval
    data_num: int = all_args.data_num
    data_maker = FCNwCARADataMaker()
    data_maker.create_artificial_olhcvs(
        config_path,
        daily_datas_path, intraday_datas_path,
        market_id, start_index,
        daily_index_interval, intraday_index_interval,
        data_num
    )

if __name__ == "__main__":
    main(sys.argv[1:])