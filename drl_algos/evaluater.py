import pathlib
from pathlib import Path
import sys
curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
parent_path: pathlib.Path = curr_path.parents[0]
grandparent_path: pathlib.Path = curr_path.parents[1]
sys.path.append(str(parent_path))
sys.path.append(str(grandparent_path))
from algorithm import Algorithm
from envs.environments import AECEnv4HeteroRL
from flex_processors import FlexProcessor
from logs import FlexSaver
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from ots import DDEvaluater
from stylized_facts import StylizedFactsChecker
import torch
from tqdm import tqdm
from typing import Any
from typing import Optional
from typing import TypeVar
import warnings
warnings.filterwarnings("ignore")

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class Evaluater:
    """Evaluater class."""
    def __init__(
        self,
        dd_evaluaters: list[DDEvaluater],
        num_points: int,
        env: AECEnv4HeteroRL,
        algo: Algorithm,
        actor_load_path: Path,
        txts_save_path: Path,
        tick_dfs_save_path: Path,
        ohlcv_dfs_save_path: Path,
        transactions_path: Path,
        session1_transactions_file_name: str,
        session2_transactions_file_name: str,
        market_name: str,
        decision_histories_save_path: Path,
        figs_save_path: Path,
        indicators_save_path: Path,
    ) -> None:
        """initialization.
        
        Args:
            env (AECEnv4HeteroRL): Environment.
            algo (Algorithm): RL algorithm.
            actor_load_path (Path): Path to load actor's weights.
            logs_save_path (Path): Path to save logs.
            market_name (str): Market name. See the config file.
            decision_histories_save_path (Path): Path to save decision histories.
            figs_save_path (Path): Path to save figures.
            indicators_save_path (Path): Path to save indicators.
        """
        self.dd_evaluaters: list[DDEvaluater] = dd_evaluaters
        if len(dd_evaluaters) == 0:
            raise ValueError("specify at least 1 DDEvaluater.")
        self.num_points: int = num_points
        self.real_tickers: list[int] = list(dd_evaluaters[0].ticker_path_dic.keys())
        self.env: AECEnv4HeteroRL = env
        self.algo: Algorithm = algo
        if actor_load_path.exists():
            self.algo.actor.load_state_dict(
                torch.load(
                    str(actor_load_path), map_location="cpu"
                )["actor_state_dict"]
            )
        else:
            raise FileNotFoundError(f"{actor_load_path} does not exist.")
        self.txts_save_path: Path = txts_save_path
        self.tick_dfs_save_path: Path = tick_dfs_save_path
        self.ohlcv_dfs_path: Path = ohlcv_dfs_save_path
        self.transactions_path: Path = transactions_path
        self.session1_transactions_file_name: str = session1_transactions_file_name
        self.session2_transactions_file_name: str = session2_transactions_file_name
        self.market_name: str = market_name
        self.decision_histories_save_path: Path = decision_histories_save_path
        self.figs_save_path: Path = figs_save_path
        self.indicators_save_path: Path = indicators_save_path

    def save_multiple_episodes(
        self,
        start_num: int = 0,
        episode_num: int = 100
    ) -> None:
        """Conduct multiple episodes and save results.
        
        Args:
            start_num (int, optional): Start number. Default to 0.
            episode_num (int, optional): Number of episodes. Default to 100.
        """
        for episode in tqdm(range(start_num, start_num + episode_num)):
            self.save_1episode(save_name=str(episode))
        processor: FlexProcessor = FlexProcessor(
            txt_datas_path=self.txts_save_path,
            csv_datas_path=self.tick_dfs_save_path,
        )
        processor.convert_all_txt2csv(
            is_bybit_format=False, is_display_path=False
        )
        checker: StylizedFactsChecker = StylizedFactsChecker(
            ohlcv_dfs_path=None,
            resample_mid=True,
            tick_dfs_path=self.tick_dfs_save_path,
            ohlcv_dfs_save_path=self.ohlcv_dfs_path,
            figs_save_path=self.figs_save_path,
            specific_name=None,
            resample_rule="1min",
            is_real=False,
            transactions_folder_path=self.transactions_path,
            session1_transactions_file_name=self.session1_transactions_file_name,
            session2_transactions_file_name=self.session2_transactions_file_name
        )
        save_path: Path = self.indicators_save_path / "stylized_facts.csv"
        checker.check_stylized_facts(save_path=save_path)
        for dd_evaluater in self.dd_evaluaters:
            dd_evaluater.add_ticker_path(
                ticker="temp", path=self.ohlcv_dfs_path
            )
        try:
            statistics: list[float] = []
            ot_distances: list[float] = []
            for dd_evaluater in self.dd_evaluaters:
                art_point_cloud, statistics_ = dd_evaluater.get_point_cloud_from_ticker(
                    ticker="temp", num_points=self.num_points,
                    save2dic=False, return_statistics=True
                )
                statistics.extend(statistics_)
                for ticker in self.real_tickers:
                    real_point_cloud: ndarray = dd_evaluater.get_point_cloud_from_ticker(
                        ticker=ticker, num_points=self.num_points, save2dic=True
                    )
                    ot_distance: float = dd_evaluater.calc_ot_distance(
                        art_point_cloud, real_point_cloud, is_per_bit=True
                    )
                    ot_distances.append(ot_distance)
                print(dd_evaluater)
                print(f"statistics: {statistics_}")
                print(f"ot_distances: {ot_distances}")
            dd_results: list[int | float] = statistics + ot_distances
        except Exception as e:
            print(e)

    def save_1episode(
        self,
        save_name: str,
    ) -> None:
        """Conduct 1 episode and save results.
        
        Args:
            save_name (str): Save name.
        """
        if isinstance(self.env.logger, FlexSaver):
            self.env.logger.txt_save_folder_path = self.txts_save_path
            self.env.logger.txt_file_name_dic[self.market_name] = f"{save_name}.txt"
        else:
            session1_start_time, session1_end_time, session2_start_time = \
                self._get_session_boundary(self.env.config_dic)
            saver: FlexSaver = FlexSaver(
                session1_end_time=session1_end_time,
                session2_start_time=session2_start_time,
                txt_save_folder_path=self.txts_save_path,
                txt_file_name_dic={self.market_name: f"{save_name}.txt"}
            )
            self.env.logger = saver
        self.env.reset()
        done: bool = False
        for agent_id in self.env.agent_iter():
            obs: ObsType = self.env.last()
            t: int = self.env.get_time()
            if t <= session1_start_time:
                action: ActionType = self.algo.explore(obs)
            else:
                action: ActionType = self.algo.exploit(obs)
            reward, done, info = self.env.step(action)
            if done:
                break
        self.create_decision_histories(save_name)

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
                session1_start_time = total_times
                session1_end_time = total_times + session_iteration_steps - 1
            elif session_name == "2":
                session2_start_time = total_times - 1
            total_times += session_iteration_steps
        if session1_end_time is None:
            raise ValueError("failed to find session 1.")
        if session2_start_time is None:
            raise ValueError("failed to found session 2.")
        return session1_end_time, session2_start_time

    def create_decision_histories(self, save_name) -> None:
        """Create decision histories df. This method is called after 1 episode."""
        obs_dic: dict[str, list[float | int]] = self.env.obs_dic
        obs_df: DataFrame = pd.DataFrame(obs_dic)
        action_dic: dict[str, list[float | int]] = self.env.action_dic
        action_df: DataFrame = pd.DataFrame(action_dic)
        reward_dic: dict[str, list[float | int]] = self.env.reward_dic
        reward_df: DataFrame = pd.DataFrame(reward_dic)
        decision_histories_df: DataFrame = pd.merge(
            obs_df, action_df, on=["step", "agent_id"]
        )
        decision_histories_df = pd.merge(
            decision_histories_df, reward_df, on=["step", "agent_id"]
        )
        decision_histories_df.set_index("step", inplace=True)
        save_path: Path = self.decision_histories_save_path / f"{save_name}.csv"
        decision_histories_df.to_csv(save_path)

