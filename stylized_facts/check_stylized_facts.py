import argparse
from matplotlib import pyplot as plt
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
from rich import print
import sys
sys.path.append(str(parent_path))
from stylized_facts import StylizedFactsChecker
from typing import Optional
plt.rcParams["font.size"] = 20

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ohlcv_folder_path", type=str, default=None,
                        help="folder path that target OHLCV datas are stored.")
    parser.add_argument("--tick_folder_path", type=str, default=None,
                        help="folder path that target tick datas are stored.")
    parser.add_argument("--resample_rule", type=str, default="1min")
    parser.add_argument("--resample_mid", action="store_true")
    parser.add_argument("--is_real", action="store_true")
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--new_ohlcv_folder_path", type=str, default=None,
                        help="folder path that target csv datas are stored.")
    parser.add_argument(
        "--transactions_folder_path", type=str, default=None,
        help="folder path that number of transaction data is stored. These are used to assign calender time to artificial data. Confirm that is_real=False."
    )
    parser.add_argument(
        "--transactions_save_folder_path", type=str, default=None,
        help="folder path that number of transaction data will be stored. This is used to save transactions of real data. Confirm that is_real=True."
    )
    parser.add_argument("--session1_transactions_file_name", type=str)
    parser.add_argument("--session2_transactions_file_name", type=str, default=None)
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
    resample_rule: str = all_args.resample_rule
    ohlcv_folder: Optional[str] = all_args.ohlcv_folder_path
    ohlcv_dfs_path: Optional[Path] = create_path(ohlcv_folder)
    tick_folder: Optional[str] = all_args.tick_folder_path
    tick_dfs_path: Optional[Path] = create_path(tick_folder)
    resample_mid: bool = all_args.resample_mid
    is_real: bool = all_args.is_real
    is_bybit: bool = all_args.is_bybit
    new_ohlcv_folder: Optional[str] = all_args.new_ohlcv_folder_path
    ohlcv_dfs_save_path: Optional[Path] = create_path(new_ohlcv_folder)
    specific_name: Optional[str] = all_args.specific_name
    choose_full_size_df: bool = all_args.choose_full_size_df
    print(f"ohlcv_dfs_path: {ohlcv_dfs_path}")
    print(f"tick_dfs_path: {tick_dfs_path}")
    print(f"is_real: {is_real} is_bybit: {is_bybit} resample_mid: {resample_mid} specific_name: {specific_name} choose_full_size_df: {choose_full_size_df}")
    print()
    figs_folder: Optional[str] = all_args.figs_folder_path
    figs_save_path: Optional[Path] = create_path(figs_folder)
    transactions_folder: Optional[str] = all_args.transactions_folder_path
    transactions_folder_path: Optional[Path] = create_path(transactions_folder)
    session1_transactions_file_name: str = all_args.session1_transactions_file_name
    session2_transactions_file_name: Optional[str] = all_args.session2_transactions_file_name
    print(f"figs_save_path: {figs_save_path} transactions_folder_path: {transactions_folder_path}")
    session1_end_time_str: Optional[str] = all_args.session1_end_time_str
    session2_start_time_str: Optional[str] = all_args.session2_start_time_str
    print(f"session1_end_time: {session1_end_time_str} session2_start_time: {session2_start_time_str}")
    print()
    checker = StylizedFactsChecker(
        seed=seed,
        ohlcv_dfs_path=ohlcv_dfs_path,
        tick_dfs_path=tick_dfs_path,
        resample_mid=resample_mid,
        resample_rule=resample_rule,
        is_real=is_real,
        is_bybit=is_bybit,
        ohlcv_dfs_save_path=ohlcv_dfs_save_path,
        choose_full_size_df=choose_full_size_df,
        specific_name=specific_name,
        figs_save_path=figs_save_path,
        transactions_folder_path=transactions_folder_path,
        session1_end_time_str=session1_end_time_str,
        session2_start_time_str=session2_start_time_str,
        session1_transactions_file_name=session1_transactions_file_name,
        session2_transactions_file_name=session2_transactions_file_name,
    )
    print(f"done! number of ohlcv dfs: {len(checker.ohlcv_dfs)} number of tick dfs: {len(checker.tick_dfs)}")
    print()
    results_folder: str = all_args.results_folder_path
    results_csv_name: str = all_args.results_csv_name
    results_save_path: Path = pathlib.Path(results_folder).resolve() / results_csv_name
    print(f"results_save_path: {results_save_path}")
    print("checking stylized facts...")
    checker.check_stylized_facts(results_save_path)
    print("done!")
    print()
    if is_real:
        transactions_save_folder: Optional[str] = all_args.transactions_save_folder_path
        transactions_save_folder_path: Optional[Path] = create_path(transactions_save_folder)
        if not transactions_save_folder_path.exists():
            transactions_save_folder_path.mkdir(parents=True)
        print(f"transactions_save_folder_path: {transactions_save_folder_path}")
        print(f"saving number of cumulative transactions...")
        checker.calc_cumulative_transactions_per_session(
            transactions_save_folder_path=transactions_save_folder_path
        )
        print("done!")
        print()
    if figs_save_path is not None:
        print("saving images to figs_save_path...")
        checker.plot_ccdf(img_save_name="ccdf.pdf")
        checker.plot_acorrs(
            lags=[lag for lag in range(1, 111)],
            img_save_name="acorrs.pdf"
        )
        checker.scatter_cumulative_transactions(
            img_save_name="transactions_time_series.pdf"
        )
        checker.plot_time_series(
            img_save_name="market_prices.pdf",
            draw_idx=42
        )
        checker.hist_features(
            img_save_name="features.pdf"
        )
        print("done!")

if __name__ == "__main__":
    main(sys.argv[1:])