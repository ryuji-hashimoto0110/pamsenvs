import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
from rich import print
import scipy.stats as stats
import sys
sys.path.append(str(parent_path))
from stylized_facts import StylizedFactsChecker
plt.rcParams["font.size"] = 20

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv_dfs_paths", type=str, nargs="*")
    parser.add_argument("--resample_rules", type=str, nargs="*")
    parser.add_argument("--labels", type=str, nargs="*")
    parser.add_argument("--colors", type=str, nargs="*")
    parser.add_argument("--fig_save_path", type=str)
    return parser

def convert_strs2paths(names: list[str]) -> list[Path]:
    paths: list[Path] =  [
        pathlib.Path(name).resolve() for name in names
    ]
    for path in paths:
        print(f"[white]{str(path)}[white]")
    print()
    return paths

def plot_ccdf_gaussian(ax: Axes, color: str):
    x: ndarray = np.linspace(1, 40, 1000)
    ccdf: ndarray = 1 - stats.norm.cdf(x)
    ax.plot(x, ccdf, color=color, label="Gaussian")
    ax.set_xscale("log")
    ax.set_yscale("log")

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    ohlcv_dfs_paths: list[Path] = convert_strs2paths(all_args.ohlcv_dfs_paths)
    resample_rules: list[str] = all_args.resample_rules
    print(f"[white]resample_rules: {resample_rules}[white]")
    labels: list[str] = all_args.labels
    print(f"[white]labels: {labels}[white]")
    colors: list[str] = all_args.colors
    print(f"[white]colors: {colors}[white]")
    fig: Figure = plt.figure(figsize=(10,6))
    ax: Axes = fig.add_subplot(1,1,1)
    for ohlcv_dfs_path, resample_rule, label, color in zip(
        ohlcv_dfs_paths, resample_rules, labels, colors
    ):
        checker = StylizedFactsChecker(
            ohlcv_dfs_path=ohlcv_dfs_path,
            resample_rule=resample_rule
        )
        checker.plot_ccdf(
            ax=ax, label=label, color=color
        )
    plot_ccdf_gaussian(ax=ax, color="red")
    plt.legend(fontsize=10)
    ax.set_xlim([1, 40])
    ax.set_ylim([1e-06, 1])
    fig_save_path: Path =  pathlib.Path(all_args.fig_save_path).resolve()
    print(f"[white]fig_save_path: {fig_save_path}[white]")
    plt.savefig(str(fig_save_path))
    
if __name__ == "__main__":
    main(sys.argv[1:])