import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from bybit_processors import BybitProcessor
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder_path", type=str, default=None,
                        help="folder path that target csv datas are stored.")
    parser.add_argument("--tickers", type=str, nargs="*")
    parser.add_argument("--start_date", type=int)
    parser.add_argument("--end_date", type=int)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    csv_folder: str = all_args.csv_folder_path
    csv_folder_path: Path = pathlib.Path(csv_folder).resolve()
    tickers: list[str] = all_args.tickers
    start_date: int = all_args.start_date
    end_date: int = all_args.end_date
    bybit_processor: BybitProcessor = BybitProcessor(csv_datas_path=csv_folder_path)
    bybit_processor.download_datas_from_bybit(tickers=tickers, start_date=start_date, end_date=end_date)

if __name__ == "__main__":
    main(sys.argv[1:])