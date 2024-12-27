import argparse
from numpy import ndarray
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
from rich import print
from rich.tree import Tree
import sys
sys.path.append(str(parent_path))
from ots import DDEvaluater
from ots import ReturnDDEvaluater
from ots import ReturnTSDDEvaluater
from ots import RVsDDEvaluater
from ots import TailReturnDDEvaluater
from typing import Optional

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ohlcv_folder_path", type=str)
    parser.add_argument("--ticker_folder_names", type=str, nargs="*", default=None)
    parser.add_argument("--ticker_file_names", type=str, nargs="*", default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--resample_rule", type=str, default="1min")
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--lags", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--point_cloud_types", type=str, nargs="+",
        choices=["return", "tail_return", "return_ts", "rv_returns"]
    )
    parser.add_argument("--distance_matrix_save_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--figs_save_path", type=str, default=None)
    parser.add_argument("--nrows_subolots", type=int, default=1)
    parser.add_argument("--ncols_subplots", type=int, default=1)
    return parser

def create_path(folder_name: Optional[str]) -> Optional[Path]:
    folder_path: Optional[Path] = None
    if folder_name is not None:
        folder_path: Path = pathlib.Path(folder_name).resolve()
    parent_path: Path = folder_path.parent
    if not parent_path.exists():
        parent_path.mkdir(parents=True)
    return folder_path

def create_ddevaluaters(all_args, show_args: bool = True) -> list[DDEvaluater]:
    """
    all_args must contain:
        - seed: int
    """
    seed: int = all_args.seed
    datas_path: str = pathlib.Path(all_args.ohlcv_folder_path).resolve()
    tree: Tree = Tree(str(datas_path))
    tickers: list[str | int] = all_args.tickers
    ticker_folder_names: list[str] = all_args.ticker_folder_names
    ticker_file_names: list[str] = all_args.ticker_file_names
    if (
        ticker_folder_names is not None and
        ticker_file_names is None
    ):
        assert len(tickers) == len(ticker_folder_names)
    elif (
        ticker_file_names is not None and
        ticker_folder_names is None
    ):
        assert len(tickers) == len(ticker_file_names)
    else:
        raise ValueError(
            "Please specify either ticker_folder_names or ticker_file_names."
        )
    ticker_path_dic: dict[str | int, Path] = {}
    target_names: list[str] = (
        ticker_folder_names if ticker_folder_names is not None else ticker_file_names
    )
    for ticker, target_name in zip(tickers, target_names):
        tree.add(f"{ticker}: {target_name}")
        ticker_path: Path = datas_path / target_name
        ticker_path_dic[ticker] = ticker_path
    print("Tickers and their corresponding paths:")
    print(tree)
    resample_rule: str = all_args.resample_rule
    is_bybit: bool = all_args.is_bybit
    point_cloud_types: list[str] = all_args.point_cloud_types
    print(f"resample_rule: {resample_rule} is_bybit: {is_bybit} point_cloud_type: {point_cloud_types}")
    evaluaters: list[DDEvaluater] = []
    if "return" in  point_cloud_types:
        evaluater: DDEvaluater = ReturnDDEvaluater(
            seed=seed, resample_rule=resample_rule, 
            is_bybit=is_bybit, ticker_path_dic=ticker_path_dic
        )
        evaluaters.append(evaluater)
    if "tail_return" in  point_cloud_types:
        evaluater: DDEvaluater = TailReturnDDEvaluater(
            seed=seed, resample_rule=resample_rule,
            is_bybit=is_bybit, ticker_path_dic=ticker_path_dic
        )
        evaluaters.append(evaluater)
    if "rv_returns" in point_cloud_types:
        evaluater: DDEvaluater = RVsDDEvaluater(
            seed=seed, resample_rule=resample_rule, 
            is_bybit=is_bybit, ticker_path_dic=ticker_path_dic
        )
        evaluaters.append(evaluater)
    if "return_ts" in point_cloud_types:
        lags: list[int] = all_args.lags
        print(f"lags: {lags}")
        if 2 <= len(lags):
            evaluater: DDEvaluater = ReturnTSDDEvaluater(
                lags=lags,
                seed=seed, resample_rule=resample_rule,
                is_bybit=is_bybit, ticker_path_dic=ticker_path_dic
            )
            evaluaters.append(evaluater)
        for lag in lags:
            evaluater: DDEvaluater = ReturnTSDDEvaluater(
                lags=[int(lag)],
                seed=seed, resample_rule=resample_rule,
                is_bybit=is_bybit, ticker_path_dic=ticker_path_dic
            )
            evaluaters.append(evaluater)
    return evaluaters

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    tickers: list[str | int] = all_args.tickers
    evaluater: DDEvaluater = create_ddevaluaters(all_args)[0]
    distance_matrix_save_path: Optional[Path] = create_path(all_args.distance_matrix_save_path)
    n_samples: int = all_args.n_samples
    print(f"Distance matrix will be saved at {str(distance_matrix_save_path)}")
    print(f"Number of samples: {n_samples}")
    distance_matrix, pvalue_matrix = evaluater.create_ot_distance_matrix(
        n_samples, tickers, distance_matrix_save_path,
        return_distance_matrix=True, return_pvalue_matrix=True
    )
    figs_save_path: Optional[Path] = create_path(all_args.figs_save_path)
    if figs_save_path is not None and distance_matrix is not None:
        tree: Tree = Tree(str(figs_save_path))
        fig_distance_matrix_name: str = "distance_matrix.pdf"
        fig_pvalue_matrix_name: str = "pvalue_matrix.pdf"
        tree.add(fig_distance_matrix_name)
        tree.add(fig_pvalue_matrix_name)
        fig_distance_matrix_path: Path = figs_save_path / fig_distance_matrix_name
        fig_pvalue_matrix_path: Path = figs_save_path / fig_pvalue_matrix_name
        fig_point_clouds_name: str = "point_clouds.pdf"
        tree.add(fig_point_clouds_name)
        fig_point_clouds_path: Path = figs_save_path / fig_point_clouds_name
        print(f"Figures will be saved at:")
        print(tree)
        evaluater.draw_distance_matrix(
            tickers, distance_matrix, fig_distance_matrix_path
        )
        evaluater.draw_distance_matrix(
            tickers, pvalue_matrix, fig_pvalue_matrix_path, title="p-value matrix"
        )
        nrow_subplots: int = all_args.nrows_subolots
        ncol_subplots: int = all_args.ncols_subplots
        if isinstance(evaluater, TailReturnDDEvaluater):
            evaluater.draw_points(
                tickers, n_samples, xlim=[0, 2], ylim=[0,70],
                xlabel="Abs Log-return ratio (Top 5%)", ylabel="Frequency",
                is_all_in_one_subplot=False, save_path=fig_point_clouds_path,
                subplots_arrangement=(nrow_subplots, ncol_subplots)
            )
        elif isinstance(evaluater, ReturnTSDDEvaluater):
            lags: list[int] = evaluater.lags
            if len(lags) == 1:
                lag: int = lags[0]
                evaluater.draw_points(
                    tickers, n_samples, xlim=[0, 20], ylim=[0, 20],
                    xlabel="Abs Log-return", ylabel=f"Abs Log-return (lag={lag})",
                    is_all_in_one_subplot=False, save_path=fig_point_clouds_path,
                    subplots_arrangement=(nrow_subplots, ncol_subplots)
                )
        elif isinstance(evaluater, ReturnDDEvaluater):
            evaluater.draw_points(
                tickers, n_samples, xlim=[-10, 10], ylim=[0,70],
                xlabel="Log-return", ylabel="Frequency",
                is_all_in_one_subplot=False, save_path=fig_point_clouds_path,
                subplots_arrangement=(nrow_subplots, ncol_subplots)
            )

if __name__ == "__main__":
    main(sys.argv[1:])