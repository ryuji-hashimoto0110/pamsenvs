import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from envs.agents import CARAFCNAgent
from flex_processors import FlexProcessor
import json
from logs import FlexSaver
from pams.runners.base import Runner
from pams.runners.sequential import SequentialRunner
import random
from rich import print
from rich.tree import Tree
from stylized_facts import StylizedFactsChecker
from tqdm import tqdm
from typing import Any
from typing import Optional
from typing import TypeVar

MarketName = TypeVar("MarketName")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--configs_folder_path", type=str)
    parser.add_argument(
        "--config_names", type=str, nargs="*",
        help="Names of simulation config files. Files must be in subordinate of configs_folder_path. " +
        "Ex: '0.config 1.config'"
    )
    parser.add_argument(
        "--txt_save_folder_paths", type=str, nargs="*",
        help="Names of folder paths to store simulation results with FLEX format. " +
        "Ex: '../../datas/artificial_datas/flex_txt/0 ../../datas/artificial_datas/flex_txt/1'"
    )
    parser.add_argument("--num_simulations", type=int, default=1000)
    parser.add_argument(
        "--tick_dfs_folder_paths", type=str, nargs="*",
        help="Names of folder paths to store structured simulation results with csv. " +
        "Ex: '../../datas/artificial_datas/flex_csv/0 ../../datas/artificial_datas/flex_csv/1'"
    )
    parser.add_argument(
        "--ohlcv_dfs_folder_paths", type=str, nargs="*",
        help="Names of folder paths to store preprocessed simulation results with OHLCV csv data. " +
        "Ex: '../../datas/artificial_datas/intraday/flex_ohlcv/1min/0 ../../datas/artificial_datas/intraday/flex_ohlcv/1min/1'"
    )
    parser.add_argument(
        "--figs_save_paths", type=str, nargs="*",
        help="Names of folder paths to store figures drawn by StylizedFactsChecker. " +
        "Ex '../../imgs/0 ../../imgs/1'"
    )
    parser.add_argument(
        "--results_save_paths", type=str, nargs="*",
        help="Names of file paths to store results by StylizedFactsChecker with csv format. " + 
        "Ex '../../stylized_facts/results/0.csv ../../stylized_facts/results/1.csv'"
    )
    parser.add_argument("--resample_rule", type=str)
    return parser

def get_config_paths(
    configs_folder_path: Path,
    config_names: list[str]
) -> list[Path]:
    config_paths: list[Path] = []
    tree: Tree = Tree(str(configs_folder_path.resolve()))
    for config_name in config_names:
        config_path: Path = configs_folder_path / config_name
        if not config_path.exists():
            raise ValueError(
                f"config path: {config_path} does not exist."
            )
        config_paths.append(config_path)
        tree.add(config_name)
    print(tree)
    print()
    return config_paths

def convert_strs2paths(names: list[str]) -> list[Path]:
    paths: list[Path] =  [
        pathlib.Path(name).resolve() for name in names
    ]
    for path in paths:
        print(str(path))
    print()
    return paths

def get_session_boundary(config: dict[str, Any]) -> tuple[int, int]:
    session_configs: list[dict[str, Any]] = config["sessions"]
    session1_end_time: Optional[int] = None
    session2_start_time: Optional[int] = None
    total_times: int = 0
    for session_config in session_configs:
        session_name: str = str(session_config["sessionName"])
        session_iteration_steps: int = int(session_config["iterationSteps"])
        if session_name == "1":
            session1_end_time = total_times + session_iteration_steps - 1
        elif session_name == "2":
            session2_start_time = total_times - 1
        total_times += session_iteration_steps
    if session1_end_time is None:
        raise ValueError("failed to find session 1.")
    if session2_start_time is None:
        raise ValueError("failed to found session 2.")
    print(f"session1_end_time: {session1_end_time} session2_start_time: {session2_start_time}")
    return session1_end_time, session2_start_time

def get_txt_file_name_dic(config: dict[str, Any], simulation_id: int) -> dict[MarketName, str]:
    market_names: list[MarketName] = config["markets"]
    txt_file_name_dic: dict[MarketName, str] = {}
    for market_name in market_names:
        txt_file_name_dic[market_name] = f"{market_name}_{simulation_id}.txt"
    return txt_file_name_dic

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    prng: random.Random = random.Random(initial_seed)
    print(f"initial_seed: {initial_seed}")
    print()
    print("[red]==get configuration paths==[red]")
    configs_folder_path: Path = pathlib.Path(all_args.configs_folder_path).resolve()
    config_names: list[str] = all_args.config_names
    config_paths: Path = get_config_paths(configs_folder_path, config_names)
    print("[red]==get folder paths to save txt data with FLEX format==[red]")
    txt_save_folder_paths: list[Path] = convert_strs2paths(all_args.txt_save_folder_paths)
    print("[red]==get folder paths to save csv data recorded for each event==[red]")
    tick_dfs_folder_paths: list[Path] = convert_strs2paths(all_args.tick_dfs_folder_paths)
    print("[red]==get folder paths to save csv data with OHLCV format==[red]")
    ohlcv_dfs_folder_paths: list[Path] = convert_strs2paths(all_args.ohlcv_dfs_folder_paths)
    print("[red]==get paths to refer to for transactions records per session==[red]")
    transactions_folder_path: Path = pathlib.Path(all_args.transactions_folder_path).resolve()
    tree: Tree = str(transactions_folder_path)
    session1_transactions_file_name: str = all_args.session1_transactions_file_name
    tree.add(session1_transactions_file_name)
    session2_transactions_file_name: str = all_args.session2_transactions_file_name
    tree.add(session2_transactions_file_name)
    print(tree)
    print()
    print("[red]==get folder paths to save figures by StylizedFactsChecker==[red]")
    figs_save_paths: list[Path] = convert_strs2paths(all_args.figs_save_paths)
    print("[red]==get file paths to save results by StylizedFactsChecker with csv.==[red]")
    results_save_paths: list[Path] = convert_strs2paths(all_args.results_save_paths)
    num_simulations: int = all_args.num_simulations
    resample_rule: str = all_args.resample_rule
    for i, config_path in enumerate(config_paths):
        print("[red]==start simulations==[red]")
        txt_save_folder_path: Path = txt_save_folder_paths[i]
        print(f"config{i+1}: {str(config_path)} results will be saved to >> {str(txt_save_folder_path)}")
        config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
        session1_end_time, session2_start_time = get_session_boundary(config)
        for simulation_id in tqdm(range(num_simulations)):
            txt_file_name_dic: dict[MarketName, str] = get_txt_file_name_dic(config, simulation_id)
            saver = FlexSaver(
                session1_end_time=session1_end_time,
                session2_start_time=session2_start_time,
                txt_save_folder_path=txt_save_folder_path,
                txt_file_name_dic=txt_file_name_dic
            )
            runner: Runner = SequentialRunner(
                settings=config,
                prng=random.Random(initial_seed+simulation_id),
                logger=saver
            )
            runner.class_register(CARAFCNAgent)
            runner.main()
        print()
        print("[red]==start processing==[red]")
        tick_dfs_folder_path: Path = tick_dfs_folder_paths[i]
        print(f"results will be saved to >> {str(tick_dfs_folder_path)}")
        processor = FlexProcessor(
            txt_datas_path=txt_save_folder_path,
            csv_datas_path=tick_dfs_folder_path
        )
        processor.convert_all_txt2csv(is_display_path=False)
        print()
        print("[red]==start checking stylized facts==[red]")
        ohlcv_dfs_folder_path: Path = ohlcv_dfs_folder_paths[i]
        figs_save_path: Path = figs_save_paths[i]
        results_save_path: Path = results_save_paths[i]
        checker = StylizedFactsChecker(
            seed=initial_seed,
            tick_dfs_path=tick_dfs_folder_path,
            resample_rule=resample_rule,
            ohlcv_dfs_save_path=ohlcv_dfs_folder_path,
            figs_save_path=figs_save_path,
            transactions_folder_path=transactions_folder_path,
            session1_transactions_file_name=session1_transactions_file_name,
            session2_transactions_file_name=session2_transactions_file_name
        )
        checker.check_stylized_facts(save_path=results_save_path)
        checker.plot_ccdf(img_save_name="ccdf.pdf")
        checker.scatter_cumulative_transactions(
            img_save_name="transactions_time_series.pdf"
        )

if __name__ == "__main__":
    main(sys.argv[1:])
