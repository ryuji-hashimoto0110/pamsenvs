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
    parser.add_argument("--ohlcv_folder_path", type=str,
                        help="folder path that target csv datas are stored. wither OHLCV or FLEX format is allowed.")
    parser.add_argument("--new_ohlcv_folder_path", type=str, default=None,
                        help="folder path that target csv datas are stored.")
    parser.add_argument("--transactions_save_path", type=str, default=None)
    parser.add_argument("--specific_name", type=str, default=None,
                        help="the specific name contained in target csv file name in common.")
    parser.add_argument("--choose_full_size_df", action="store_true")
    parser.add_argument("--need_resample", action="store_true")
    parser.add_argument("--figs_folder_path", type=str, default=None)
    parser.add_argument("--results_folder_path", type=str)
    parser.add_argument("--results_csv_name", type=str)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    ohlcv_folder: str = all_args.ohlcv_folder_path
    ohlcv_dfs_path: Path = pathlib.Path(ohlcv_folder).resolve()
    new_ohlcv_folder: Optional[str] = all_args.new_ohlcv_folder_path
    if new_ohlcv_folder is not None:
        ohlcv_dfs_save_path: Optional[Path] = pathlib.Path(new_ohlcv_folder).resolve()
    else:
        ohlcv_dfs_save_path = None
    specific_name: Optional[str] = all_args.specific_name
    choose_full_size_df: bool = all_args.choose_full_size_df
    need_resample: bool = all_args.need_resample
    figs_folder: Optional[str] = all_args.figs_folder_path
    if figs_folder is not None:
        figs_save_path: Optional[Path] = pathlib.Path(figs_folder).resolve()
    else:
        figs_save_path = None
    checker = StylizedFactsChecker(
        ohlcv_dfs_path=ohlcv_dfs_path,
        ohlcv_dfs_save_path=ohlcv_dfs_save_path,
        choose_full_size_df=choose_full_size_df,
        specific_name=specific_name,
        need_resample=need_resample,
        figs_save_path=figs_save_path
    )
    results_folder: str = all_args.results_folder_path
    results_csv_name: str = all_args.results_csv_name
    results_save_path: Path = pathlib.Path(results_folder).resolve() / results_csv_name
    checker.check_stylized_facts(results_save_path)
    if figs_save_path is not None:
        transactions_save_path: Path = pathlib.Path(all_args.transactions_save_path)
        transactions_folder_path: Path = transactions_save_path.parent
        if not transactions_folder_path.exists():
            transactions_folder_path.mkdir(parents=True)
        checker.plot_ccdf(img_save_name="ccdf.pdf")
        checker.scatter_cumulative_transactions(
            img_save_name="transactions_time_series.pdf",
            transactions_save_path=transactions_save_path
        )

if __name__ == "__main__":
    main(sys.argv[1:])