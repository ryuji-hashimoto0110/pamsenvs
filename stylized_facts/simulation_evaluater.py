import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
import copy
from datetime import date
from datetime import timedelta
from envs.agents import aFCNAgent
from envs.agents import CARAFCNAgent
from envs.agents import MoodAwareCARAFCNAgent
from envs.markets import MoodAwareMarket
from envs.markets import TotalTimeAwareMarket
from envs.markets import YesterdayAwareMarket
from envs.runners import SimulatorGivenRunner
from flex_processors import FlexProcessor
import json
from logs import FlexSaver
from pams.runners import Runner
from pams.runners import SequentialRunner
from pams.simulator import Simulator
from ohlcv_processors import OHLCVProcessor
import random
from rich import print
from rich.tree import Tree
from .stylized_facts_checker import StylizedFactsChecker
import subprocess
from tqdm import tqdm
from typing import Any
from typing import Optional
from typing import TypeVar
import warnings

MarketName = TypeVar("MarketName")

class SimulationEvaluater:
    """SimulationEvaluater class.

    SimulationEvaluater provides multiple methods to
        - conduct multiple simulations and save the results with FLEX format.
        - check stylized facts.
    """
    def __init__(
        self,
        initial_seed: int,
        show_process: bool = True,
        significant_figures: int = 10,
        config: Optional[dict[str, Any]] = None,
        config_path: Optional[Path | str] = None,
        specific_name: Optional[str] = None,
        txts_path: Optional[Path | str] = None,
        resample_rule: str = "1min",
        resample_mid: bool = True,
        tick_dfs_path: Optional[Path | str] = None,
        ohlcv_dfs_path: Optional[Path | str] = None,
        all_time_ohlcv_dfs_path: Optional[Path | str] = None,
        transactions_path: Optional[Path | str] = None,
        session1_transactions_file_name: Optional[str] = None,
        session2_transactions_file_name: Optional[str] = None,
        figs_save_path: Optional[Path | str] = None,
        results_save_path: Optional[Path | str] = None
    ) -> None:
        """initialization.

        Args:
            initial_seed (int): seed number to start first simulation.
            show_process (bool): whether to print process.
            significant_figures (int): significant figures for logger.
            config (dict, optional): configuration dictionary.
            config_path (Path | str, optional): path to configuration file.
                Ex) 'pamsenvs/examples/asymmetric_volatility/volatility_feedback_alpha010.json'
            specific_name (str, Optional): specific name contained in all stored file names.
            txts_path (Path | str, optional): folder path to store simulation results with FLEX format.
                Ex) 'pamsenvs/datas/artificial_datas/flex_txt/asymmetric_volatility/volatility_feedback_alpha010'
            resample_rule (str): resample frequency.
            resample_mid (bool): whether to resample mid price.
            tick_dfs_path (Path | str, optional): folder path to store executions csvs.
                Ex) 'pamsenvs/datas/artificial_datas/flex_csv/asymmetric_volatility/volatility_feedback_alpha010'
            ohlcv_dfs_path (Path | str, optional): folder path to store preprocessed OHLCV csvs.
                Ex) 'pamsenvs/datas/artificial_datas/intraday/flex_ohlcv/1min/asymmetric_volatility/volatility_feedback_alpha010'
            all_time_ohlcv_dfs_path (Path | str, optional)
            transactions_path (Path | str, optional): folder path to load transactions data. equivalent to transactions_folder_path in StylizedFactsCheker.
                Ex) 'pamsenvs/datas/real_datas/intraday/flex_transactions/1min/9202'
            session1_transactions_file_name (str, optional): file name of cumulative transactions data in session 1.
            session2_transactions_file_name (str, optional): file name of cumulative transactions data in session 2.
            figs_save_path (Path | str, optional): folder path to store figures.
                Ex)'pamsenvs/imgs/asymmetric_volatility/volatility_feedback_alpha010
            results_save_path (Path | str, optional): csv file path to store results by StylizedFactsChecker.
        """
        self.initial_seed: int = initial_seed
        self.show_process: bool = show_process
        self.significant_figures: int = significant_figures
        self.config: Optional[dict[str, Any]] = config
        self.config_path: Optional[Path] = self._convert_str2path(config_path, mkdir=False)
        self.specific_name: Optional[str] = specific_name
        self.txts_path: Optional[Path] = self._convert_str2path(txts_path, mkdir=True)
        self.resample_rule: str = resample_rule
        self.resample_mid: bool = resample_mid
        self.tick_dfs_path: Optional[Path] = self._convert_str2path(tick_dfs_path, mkdir=True)
        self.ohlcv_dfs_path: Optional[Path] = self._convert_str2path(ohlcv_dfs_path, mkdir=True)
        self.all_time_ohlcv_dfs_path: Optional[Path] = self._convert_str2path(all_time_ohlcv_dfs_path, mkdir=True)
        self.transactions_path: Optional[Path] = self._convert_str2path(transactions_path, mkdir=False)
        self.session1_transactions_file_name: Optional[str] = session1_transactions_file_name
        self.session2_transactions_file_name: Optional[str] = session2_transactions_file_name
        self.figs_save_path: Optional[Path] = self._convert_str2path(figs_save_path, mkdir=True)
        self.results_save_path: Optional[Path] = self._convert_str2path(results_save_path, mkdir=False)

    def _convert_str2path(
        self,
        path_name: Optional[str | Path],
        mkdir: bool
    ) -> Optional[Path]:
        """convert str to path.
        
        Args:
            path_name (str | Path, optional): name of path. If path_name is str, it will be
                converted to Path. path_name is recommended to be an absolute path.
            mkdir (bool): whether to make directory if path does not exist.
        """
        if path_name is None:
            return None
        elif isinstance(path_name, Path):
            path: Path = path_name.resolve()
        elif isinstance(path_name, str):
            path: Path = pathlib.Path(path_name).resolve()
        else:
            raise ValueError(
                f"path_name must be either str or pathlib.Path. path_name={path_name}"
            )
        if not path.exists():
            if mkdir:
                path.mkdir(parents=True)
        return path        
    
    def _get_session_boundary(self, config: dict[str, Any]) -> tuple[int, int]:
        """get session boundary.

        Args:
            config (dict): configuration dictionary.
        
        Returns:
            session1_end_time (int): time step to end session 1.
            session2_start_time (int): time step to start session 2.
        """
        session_configs: list[dict[str, Any]] = config["simulation"]["sessions"]
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
        return session1_end_time, session2_start_time
    
    def _get_txt_file_name_dic(
        self,
        config: dict[str, Any],
        today_date: date
    ) -> dict[MarketName, str]:
        """get txt file names to store simulation results.

        The name of txt file is made of specific name, market name, and the pseudo date.

        Args:
            simulation_id (int): simulation ID.
            today_date (date): date.
        """
        market_names: list[MarketName] = config["simulation"]["markets"]
        if self.specific_name is None:
            raise ValueError("specify specific_name.")
        if len(market_names) != 1:
            raise NotImplementedError(
                "multiple markets are not implemented."
            )
        txt_file_name_dic: dict[MarketName, str] = {}
        today_date_str: str = today_date.strftime(format='%Y%m%d')
        for market_name in market_names:
            txt_file_name_dic[market_name] = f"{self.specific_name}_{market_name}_{today_date_str}.txt"
        return txt_file_name_dic
    
    def _class_register(self, runner: Runner) -> None:
        """register classes."""
        runner.class_register(aFCNAgent) if aFCNAgent not in runner.registered_classes else ...
        runner.class_register(CARAFCNAgent) if CARAFCNAgent not in runner.registered_classes else ...
        runner.class_register(MoodAwareCARAFCNAgent) \
            if MoodAwareCARAFCNAgent not in runner.registered_classes else ...
        runner.class_register(MoodAwareMarket) if MoodAwareMarket not in runner.registered_classes else ...
        runner.class_register(TotalTimeAwareMarket) \
            if TotalTimeAwareMarket not in runner.registered_classes else ...
        runner.class_register(YesterdayAwareMarket) \
            if YesterdayAwareMarket not in runner.registered_classes else ...

    def simulate_multiple_times(
        self,
        num_simulations: int,
        use_simulator_given_runner: bool,
        start_date: Optional[date] = None
    ) -> tuple[date, date]:
        """conduct simulations multiple times.
        
        Args:
            num_simulation (int): number of simulations.
            use_simulator_given_runner (bool): whether to use SimulatorGivenRunner.
            start_date (date, optional): the starting pseudo date assigned to the txt files of the simulation results.
                Default to None.
        
        Returns:
            start_date (date): start date
            end_date (date): end date
        """
        if self.txts_path is None:
            raise ValueError("spevify txts_path.")
        if self.config is not None:
            config: dict[str, Any] = self.config
        elif self.config_path is not None:
            config: dict[str, Any] = json.load(fp=open(str(self.config_path), mode="r"))
        else:
            raise ValueError("specify config or config_path.")
        session1_end_time, session2_start_time = self._get_session_boundary(config)
        today_date: date = date(year=2015, month=1, day=1) if start_date is None else start_date
        start_date: date = today_date
        previous_simulator: Optional[Simulator] = None
        pending_simulator: Optional[Simulator] = None
        if use_simulator_given_runner:
            runner: Runner = SimulatorGivenRunner(settings=config)
        exceptions_dic: dict[int, str] = {}
        if self.show_process:
            print("[green]==start simulations==[green]")
            print(f"config-> {str(self.config_path)}  txts-> {str(self.txts_path)}")
            print(f"Session 1 end at time {session1_end_time}. Session 2 start at time {session2_start_time}.")
            print(f"Whether to use SimulatorGivenRunner: {use_simulator_given_runner}")
        for simulation_id in tqdm(range(num_simulations)):
            txt_file_name_dic: dict[MarketName, str] = self._get_txt_file_name_dic(
                config, today_date
            )
            saver = FlexSaver(
                significant_figures=self.significant_figures,
                session1_end_time=session1_end_time,
                session2_start_time=session2_start_time,
                txt_save_folder_path=self.txts_path,
                txt_file_name_dic=txt_file_name_dic
            )
            if use_simulator_given_runner:
                runner.set_seed(self.initial_seed+simulation_id)
                self._class_register(runner)
                runner._setup(
                    previous_simulator=previous_simulator,
                    new_logger=saver
                )
                pending_simulator = copy.deepcopy(runner.simulator)
            else:
                runner: Runner = SequentialRunner(
                    settings=config,
                    prng=random.Random(self.initial_seed+simulation_id),
                    logger=saver
                )
                self._class_register(runner)
                runner._setup()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    runner._run()
            except Exception as e:
                exceptions_dic[simulation_id] = e
                if use_simulator_given_runner:
                    previous_simulator = copy.deepcopy(pending_simulator)
                continue
            today_date += timedelta(days=1)
            if use_simulator_given_runner:
                del previous_simulator
                del pending_simulator
                previous_simulator = copy.deepcopy(runner.simulator)
        end_date: date = today_date
        start_date_str: str = start_date.strftime(format='%Y%m%d')
        end_date_str: str = end_date.strftime(format='%Y%m%d')
        if self.show_process:
            print("[green]==simulations ended==[green]")
            print(exceptions_dic)
            print(f"Pseudo dates are assined to each artificial data. [{start_date_str}->{end_date_str}]")
            print()
        return start_date, end_date

    def process_flex(self) -> None:
        """convert raw txt data to execution-extracted csv data."""
        if self.txts_path is None:
            raise ValueError("spevify txts_path.")
        if len(list(self.txts_path.iterdir())) == 0:
            raise ValueError("txts_path is empty. Run simulate_multiple_times first.")
        if self.tick_dfs_path is None:
            raise ValueError("spevify tick_dfs_path.")
        if 0 < len(list(self.tick_dfs_path.iterdir())):
            warnings.warn("tick_dfs_path is not empty. Some files are possible to be overwritten.")
        if self.show_process:
            print("[green]==convert txt to csv==[green]")
            print("Extract execution events from txt datas and convert them into csv datas.")
            print(f"txts-> {str(self.txts_path)} tick csvs-> {str(self.tick_dfs_path)}")
        processor = FlexProcessor(
            txt_datas_path=self.txts_path,
            csv_datas_path=self.tick_dfs_path
        )
        processor.convert_all_txt2csv(is_display_path=False)
        if self.show_process:
            print("[green]==converting process ended==[green]")
            print()

    def check_stylized_facts(
        self,
        check_stylized_facts: bool = True,
        check_asymmetry: bool = False,
        check_asymmetry_path: str = "check_asymmetry.R",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> None:
        """check stylized facts.

        Args:
            check_stylized_facts (bool): whether to check stylized facts.
            check_asymmetry (bool): whether to check asymmetric volatility.
            check_asymmetry_path (str): path to the executable file 'check_asymmetry.R'
            start_date (date, optional)
            end_date (date, optional)
        """
        if self.tick_dfs_path is None:
            raise ValueError("spevify tick_dfs_path.")
        if len(list(self.tick_dfs_path.iterdir())) == 0:
            raise ValueError("tick_dfs_path is empty. Run process_flex first.")
        if self.ohlcv_dfs_path is None:
            raise ValueError("spevify ohlcv_dfs_path.")
        if 0 < len(list(self.ohlcv_dfs_path.iterdir())):
            warnings.warn("ohlcv_dfs_path is not empty. Some files are possible to be overwritten.")
        if self.transactions_path is None:
            raise ValueError("specify transactions_path.")
        if self.session1_transactions_file_name is None:
            raise ValueError("specify session1_transactions_file_name.")
        if self.session2_transactions_file_name is None:
            raise ValueError("specify session2_transactions_file_name.")
        if self.show_process:
            print("[green]==check stylized facts==[green]")
            print(f"tick csvs-> {str(self.tick_dfs_path)} OHLCV csvs-> {str(self.ohlcv_dfs_path)}")
            print(f"specific name that must be contained in file names-> {self.specific_name}")
            print(f"figures-> {self.figs_save_path}")
            tree: Tree = Tree(str(self.transactions_path.resolve()))
            tree.add(self.session1_transactions_file_name)
            tree.add(self.session2_transactions_file_name)
            print("transaction files:")
            print(tree)
        checker = StylizedFactsChecker(
            seed=self.initial_seed,
            ohlcv_dfs_path=None,
            resample_mid=self.resample_mid,
            tick_dfs_path=self.tick_dfs_path,
            ohlcv_dfs_save_path=self.ohlcv_dfs_path,
            figs_save_path=self.figs_save_path,
            specific_name=self.specific_name,
            resample_rule=self.resample_rule,
            is_real=False,
            transactions_folder_path=self.transactions_path,
            session1_transactions_file_name=self.session1_transactions_file_name,
            session2_transactions_file_name=self.session2_transactions_file_name
        )
        if check_stylized_facts:
            if self.results_save_path is None:
                raise ValueError("specify results_save_path.")
            if self.show_process:
                print(f"results-> {str(self.results_save_path)}")
            checker.check_stylized_facts(save_path=self.results_save_path)
        if check_asymmetry:
            if start_date is None:
                raise ValueError("specify start_date.")
            if end_date is None:
                raise ValueError("specify end_date.")
            self.concat_ohlcv(start_date=start_date, end_date=end_date)
            start_date_str: str = start_date.strftime(format='%Y%m%d')
            end_date_str: str = end_date.strftime(format='%Y%m%d')
            all_time_ohlcv_df_path: Path = (
                self.all_time_ohlcv_dfs_path
                / f"{self.specific_name}_{start_date_str}_{end_date_str}.csv"
            )
            check_asymmetry_command: str = f"Rscript {check_asymmetry_path} " + \
            f"{str(all_time_ohlcv_df_path)} {self.resample_rule} {self.resample_rule-1} close"
            _ = subprocess.run(check_asymmetry_command, shell=True)
        if self.show_process:
            print("[green]==stylized facts checking process ended==[green]")
            print()

    def concat_ohlcv(self, start_date: date, end_date: date) -> None:
        if self.all_time_ohlcv_dfs_path is None:
            raise ValueError("specify all_time_ohlcv_dfs_path.")
        if self.ohlcv_dfs_path is None:
            raise ValueError("spevify ohlcv_dfs_path.")
        if len(list(self.ohlcv_dfs_path.iterdir())) == 0:
            raise ValueError("ohlcv_dfs_path is empty. Run check_stylized_facts first.")
        if self.show_process:
            print("[green]==concat daily OHLCV datas==[green]")
            print(f"daily OHLCV csvs-> {str(self.ohlcv_dfs_path)} all-time OHLCV csv-> {str(self.all_time_ohlcv_dfs_path)}")
        processor = OHLCVProcessor(
            tickers=[self.specific_name],
            daily_ohlcv_dfs_path=self.ohlcv_dfs_path,
            all_time_ohlcv_dfs_path=self.all_time_ohlcv_dfs_path,
            start_date=start_date,
            end_date=end_date
        )
        processor.concat_all_ohlcv_dfs(ticker_first=False)
