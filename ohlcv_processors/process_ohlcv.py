import argparse
from datetime import date
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
from rich import print
import sys
sys.path.append(str(parent_path))
from ohlcv_processors import OHLCVProcessor

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, nargs="*")
    parser.add_argument("--daily_ohlcv_folder_path", type=str)
    parser.add_argument("--all_time_ohlcv_folder_path", type=str)
    parser.add_argument("--start_year", type=int)
    parser.add_argument("--start_month", type=int, default=1)
    parser.add_argument("--start_day", type=int, default=1)
    parser.add_argument("--end_year", type=int)
    parser.add_argument("--end_month", type=int, default=12)
    parser.add_argument("--end_day", type=int, default=31)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    tickers: list[int] = all_args.tickers
    daily_ohlcv_folder_path: Path = pathlib.Path(
        all_args.daily_ohlcv_folder_path
    ).resolve()
    all_time_ohlcv_folder_path: Path = pathlib.Path(
        all_args.all_time_ohlcv_folder_path
    ).resolve()
    start_year: int = all_args.start_year
    start_month: int = all_args.start_month
    start_day: int = all_args.start_day
    start_date: date = date(year=start_year, month=start_month, day=start_day)
    end_year: int = all_args.end_year
    end_month: int = all_args.end_month
    end_day: int = all_args.end_day
    end_date: date = date(year=end_year, month=end_month, day=end_day)
    processor = OHLCVProcessor(
        tickers=tickers,
        daily_ohlcv_dfs_path=daily_ohlcv_folder_path,
        all_time_ohlcv_dfs_path=all_time_ohlcv_folder_path,
        start_date=start_date,
        end_date=end_date
    )
    processor.concat_all_ohlcv_dfs()

if __name__ == "__main__":
    main(sys.argv[1:])