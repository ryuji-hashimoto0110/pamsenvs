import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from stylized_facts import StylizedFactsChecker
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ohlcv_folder_path", type=str, default=None,
                        help="folder path that target OHLCV datas are stored.")
    parser.add_argument("--tick_folder_path", type=str, default=None,
                        help="folder path that target tick datas are stored.")
    parser.add_argument("--is_real", action="store_true")
    parser.add_argument("--new_ohlcv_folder_path", type=str, default=None,
                        help="folder path that target csv datas are stored.")
    parser.add_argument(
        "--transactions_folder_path", type=str, default=None,
        help="folder path that number of transaction data is stored. These are used to assign calender time to artificial data. Confirm that is_real=False."
    )
    parser.add_argument("--transactions_save_folder_path", type=str, default=None)
    parser.add_argument(
        help="folder path that number of transaction data will be stored. This is used to save transactions of real data. Confirm that is_real=True."
    )
    parser.add_argument("--specific_name", type=str, default=None,
                        help="the specific name contained in target csv file name in common.")
    parser.add_argument("--choose_full_size_df", action="store_true")
    parser.add_argument("--figs_folder_path", type=str, default=None)
    parser.add_argument("--session1_end_time_str", type=str, default=None)
    parser.add_argument("--session2_start_time_str", type=str, default=None)
    parser.add_argument("--results_folder_path", type=str)
    parser.add_argument("--results_csv_name", type=str)
    return parser

def create_path(folder_name: Optional[str]) -> Optional[Path]:
    folder_path: Optional[Path] = None
    if folder_name is not None:
        folder_path: Path = pathlib.Path(folder_name).resolve()
    return folder_path

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    seed: int = all_args.seed
    ohlcv_folder: Optional[str] = all_args.ohlcv_folder_path
    ohlcv_dfs_path: Optional[Path] = create_path(ohlcv_folder)
    tick_folder: Optional[str] = all_args.tick_folder_path
    tick_dfs_path: Optional[Path] = create_path(tick_folder)
    is_real: bool = all_args.is_real
    new_ohlcv_folder: Optional[str] = all_args.new_ohlcv_folder_path
    ohlcv_dfs_save_path: Optional[Path] = create_path(new_ohlcv_folder)
    specific_name: Optional[str] = all_args.specific_name
    choose_full_size_df: bool = all_args.choose_full_size_df
    figs_folder: Optional[str] = all_args.figs_folder_path
    figs_save_path: Optional[Path] = create_path(figs_folder)
    transactions_folder: Optional[str] = all_args.transactions_folder
    transactions_folder_path: Optional[Path] = create_path(transactions_folder)
    session1_end_time_str: Optional[str] = all_args.session1_end_time_str
    session2_start_time_str: Optional[str] = all_args.session2_start_time_str
    checker = StylizedFactsChecker(
        seed=seed,
        ohlcv_dfs_path=ohlcv_dfs_path,
        tick_dfs_path=tick_dfs_path,
        is_real=is_real,
        ohlcv_dfs_save_path=ohlcv_dfs_save_path,
        choose_full_size_df=choose_full_size_df,
        specific_name=specific_name,
        figs_save_path=figs_save_path,
        transactions_folder_path=transactions_folder_path,
        session1_end_time_str=session1_end_time_str,
        session2_start_time_str=session2_start_time_str
    )
    results_folder: str = all_args.results_folder_path
    results_csv_name: str = all_args.results_csv_name
    results_save_path: Path = pathlib.Path(results_folder).resolve() / results_csv_name
    checker.check_stylized_facts(results_save_path)
    if is_real:
        transactions_save_folder: Optional[str] = all_args.transactions_save_folder_path
        transactions_save_folder_path: Optional[Path] = create_path(transactions_save_folder)
        if not transactions_save_folder_path.exists():
            transactions_save_folder_path.mkdir(parents=True)
        checker.calc_cumulative_transactions_per_session(
            transactions_save_folder_path=transactions_save_folder_path
        )
    if figs_save_path is not None:
        checker.plot_ccdf(img_save_name="ccdf.pdf")
        checker.scatter_cumulative_transactions(
            img_save_name="transactions_time_series.pdf"
        )

if __name__ == "__main__":
    main(sys.argv[1:])