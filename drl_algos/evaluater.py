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
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from ots import DDEvaluater
import seaborn as sns
from stylized_facts import StylizedFactsChecker
import torch
from tqdm import tqdm
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeVar
import warnings
plt.rcParams["font.size"] = 20
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
        stylized_facts_save_path: Path,
        ot_distances_save_path: Path,
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
        self.env: AECEnv4HeteroRL = env
        self.algo: Algorithm = algo
        if actor_load_path.exists():
            self.save_name: str = actor_load_path.stem
            self.algo.actor.load_state_dict(
                torch.load(
                    str(actor_load_path), map_location="cuda:1"
                )["actor_state_dict"]
            )
            self.actor_configs: list[float] = self._get_actor_configs(actor_load_path)
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
        self.stylized_facts_save_path: Path = stylized_facts_save_path
        self.ot_distances_save_path: Path = ot_distances_save_path
        self.column_names: list[str] = self._get_column_names()

    def _get_actor_configs(self, actor_load_path: Path) -> list[float]:
        """get actor's configurations.

        actor's checkpoint file shoud have a name structured:
            be(la)st-{sigma_str}-{alpha_str}-{gamma_str}-{all_args.seed}.pt
        
        Args:
            actor_load_path (Path): Path to load actor's weights.
        
        Returns:
            actor_configs (list): Actor's configurations.
        """
        actor_name: str = actor_load_path.stem
        actor_name = actor_name.split("-")
        actor_configs: list[float] = []
        for config in actor_name:
            if config == "best" or config == "last":
                continue
            actor_configs.append(self._str2float(config))
        return actor_configs[:-1]
    
    def _str2float(self, s: str):
        num = int(s)
        length = len(s)
        return num / (10 ** (length - 1))
    
    def _get_column_names(self) -> list[str]:
        column_names: list[str] = [
            "sigma", "alpha", "gamma"
        ]
        for dd_evaluater in self.dd_evaluaters:
            dd_name: str = str(dd_evaluater)
            statistics_names: list[str] = dd_evaluater.get_statistics()
            ot_distance_names: list[str] = [
                f"{dd_name}({ticker})" for ticker in dd_evaluater.ticker_path_dic.keys()
            ]
            column_names.extend(statistics_names)
            column_names.extend(ot_distance_names)
        return column_names

    def save_multiple_episodes(
        self,
        start_num: int = 0,
        episode_num: int = 300,
        unlink_all: bool = False
    ) -> list[DataFrame]:
        """Conduct multiple episodes and save results.
        
        Args:
            start_num (int, optional): Start number. Default to 0.
            episode_num (int, optional): Number of episodes. Default to 100.
            unlink_all (bool, optional): Whether to unlink all files. Default to False.
        """
        decision_histories_dfs: list[DataFrame] = []
        for episode in tqdm(range(start_num, start_num + episode_num)):
            decision_histories_dfs.append(
                self.save_1episode(save_name=str(episode))
            )
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
        checker.check_stylized_facts(save_path=self.stylized_facts_save_path)
        if self.ot_distances_save_path.exists():
            dd_df: DataFrame = pd.read_csv(
                self.ot_distances_save_path, index_col=0
            )
            if len(dd_df.columns) != len(self.column_names):
                raise ValueError(
                    f"columns are different.\n" + \
                    f"dd_df.columns: {dd_df.columns}\n" + \
                    f"self.column_names: {self.column_names}"
                )
        else:
            dd_df: DataFrame = DataFrame(columns=self.column_names)
        for dd_evaluater in self.dd_evaluaters:
            dd_evaluater.add_ticker_path(
                ticker="temp", path=self.ohlcv_dfs_path
            )
        columns: list[float] = self.actor_configs
        try:
            for dd_evaluater in self.dd_evaluaters:
                real_tickers: list[int] = list(dd_evaluater.ticker_path_dic.keys())
                art_point_cloud, statistics = dd_evaluater.get_point_cloud_from_ticker(
                    ticker="temp", num_points=self.num_points,
                    save2dic=False, return_statistics=True
                )
                ot_distances: list[float] = []
                for ticker in real_tickers:
                    if ticker == "temp":
                        continue
                    real_point_cloud: ndarray = dd_evaluater.get_point_cloud_from_ticker(
                        ticker=ticker, num_points=self.num_points, save2dic=True
                    )
                    ot_distance: float = dd_evaluater.calc_ot_distance(
                        art_point_cloud, real_point_cloud, is_per_bit=True
                    )
                    ot_distances.append(ot_distance)
                columns.extend(statistics)
                columns.extend(ot_distances)
            dd_df.loc[self.save_name] = columns
        except Exception as e:
            print(e)
        dd_df.to_csv(self.ot_distances_save_path)
        if unlink_all:
            self._unlink_all()
        return decision_histories_dfs

    def _unlink_all(self) -> None:
        """Unlink all files."""
        for path in [
            self.txts_save_path, self.tick_dfs_save_path,
            self.ohlcv_dfs_path, self.decision_histories_save_path
        ]:
            if path.exists():
                for file in path.iterdir():
                    file.unlink()

    @torch.no_grad()
    def save_1episode(
        self,
        save_name: str,
    ) -> DataFrame:
        """Conduct 1 episode and save results.
        
        Args:
            save_name (str): Save name.
        """
        session1_start_time, session1_end_time, session2_start_time = \
            self._get_session_boundary(self.env.config_dic)
        if isinstance(self.env.logger, FlexSaver):
            self.env.logger.txt_save_folder_path = self.txts_save_path
            self.env.logger.txt_file_name_dic[self.market_name] = f"{save_name}.txt"
        else:
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
                action, _ = self.algo.explore(obs)
            else:
                action: ActionType = self.algo.exploit(obs)
            reward, done, info = self.env.step(action)
            if done:
                break
        decision_histories_df: DataFrame = self.create_decision_histories(save_name)
        return decision_histories_df

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
        return session1_start_time, session1_end_time, session2_start_time

    def create_decision_histories(self, save_name) -> DataFrame:
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
        decision_histories_df.sort_values(by="step", inplace=True)
        decision_histories_df.set_index("step", inplace=True)
        save_path: Path = self.decision_histories_save_path / f"{save_name}.csv"
        decision_histories_df.to_csv(save_path)
        return decision_histories_df
    
    def hist_obs_actions(
        self,
        decision_histories_dfs: list[DataFrame],
        obs_save_name: str,
        action_save_name: str,
    ) -> None:
        obs_action_dic: dict[str, list[float]] = self._get_obs_action(decision_histories_dfs)
        fig: Figure = plt.figure(figsize=(40, 20))
        for i, obs_name in enumerate(self.env.obs_names):
            ax: Axes = fig.add_subplot(3, 4, i+1)
            ax.hist(obs_action_dic[obs_name], bins=100)
            ax.set_xlabel(obs_name)
            ax.set_xlim(-1, 1)
        save_path: Path = self.figs_save_path / obs_save_name
        fig.savefig(save_path, bbox_inches="tight")
        fig: Figure = plt.figure(figsize=(15, 12))
        for i, action_name in enumerate(self.env.action_names):
            ax: Axes = fig.add_subplot(2, 1, i+1)
            ax.hist(obs_action_dic[action_name], bins=100)
            ax.set_xlabel(action_name)
            ax.set_xlim(-1, 1)
        save_path: Path = self.figs_save_path / action_save_name
        fig.savefig(save_path, bbox_inches="tight")

    def _get_obs_action(self, decision_histories_dfs: list[DataFrame]) -> dict[str, list[float]]:
        obs_action_dic: dict[str, list[float]] = {}
        for obs_name in self.env.obs_names:
            obs_action_dic[obs_name] = []
        for action_name in self.env.action_names:
            obs_action_dic[action_name] = []
        for decision_histories_df in decision_histories_dfs:
            for column_name in obs_action_dic.keys():
                obs_action_dic[column_name].extend(
                    decision_histories_df[column_name].values.tolist()
                )
        return obs_action_dic
    
    def scatter_pl_given_agent_trait(
        self,
        decision_histories_dfs: list[DataFrame],
        trait_column_names: list[
            Literal[
                "skill_boundedness", "risk_aversion_term", "discount_factor"
            ]
        ],
        save_names: list[str],
    ) -> None:
        """Scatter profit or loss given agent's trait.
        
        decision_histries_df contains:
            - step
            - agent_id
            - asset_volume
            - cash_amount
            - market_price

        The wealth of agent j at step t is given by:
            wealth_{j, t} = asset_volume_{j, t} * market_price_{t} + cash_amount_{j, t}

        The profit or loss of agent j is then given by:
            pl_j = log(wealth_{j, T}) - log(wealth_{j, 0})
        T denotes the maximum value of 'step' column of agent j.
        """
        assert len(trait_column_names) == len(save_names)
        figs: list[Figure] = [
            plt.figure(figsize=(12, 8)) for _ in range(len(trait_column_names))
        ]
        axes: list[Axes] = [fig.add_subplot(111) for fig in figs]
        for ax, trait_column_name in zip(axes, trait_column_names):
            if trait_column_name == "skill_boundedness":
                ax.set_xlabel(
                    r"skill boundedness $\sigma^j$"
                )
            elif trait_column_name == "risk_aversion_term":
                ax.set_xlabel(
                    r"risk aversion term $\alpha^j$"
                )
            elif trait_column_name == "discount_factor":
                ax.set_xlabel(
                    r"discount factor $\gamma^j$"
                )
            else:
                raise ValueError(f"trait_column_name={trait_column_name} is invalid.")
            ax.set_ylabel("profit or loss")
        for decision_histories_df in decision_histories_dfs:
            trait_dic, pls = self._get_trait_pl(decision_histories_df)
            for ax, trait_column_name in zip(axes, trait_column_names):
                trait_values: list[float] = trait_dic[trait_column_name]
                assert len(trait_values) == len(pls)
                ax.scatter(trait_values, pls, s=1, c="black")
        for fig, save_name in zip(figs, save_names):
            save_path: Path = self.figs_save_path / save_name
            fig.savefig(save_path, bbox_inches="tight")

    def scatter_price_range_given_agent_trait(
        self,
        decision_histories_dfs: list[DataFrame],
        trait_column_names: list[
            Literal[
                "skill_boundedness", "risk_aversion_term", "discount_factor"
            ]
        ],
        save_names: list[str],
    ) -> None:
        assert len(trait_column_names) == len(save_names)
        figs: list[Figure] = [
            plt.figure(figsize=(12, 8)) for _ in range(len(trait_column_names))
        ]
        axes: list[Axes] = [fig.add_subplot(111) for fig in figs]
        for ax, trait_column_name in zip(axes, trait_column_names):
            if trait_column_name == "skill_boundedness":
                ax.set_xlabel(
                    r"skill boundedness $\sigma^j$"
                )
            elif trait_column_name == "risk_aversion_term":
                ax.set_xlabel(
                    r"risk aversion term $\alpha^j$"
                )
            elif trait_column_name == "discount_factor":
                ax.set_xlabel(
                    r"discount factor $\gamma^j$"
                )
            else:
                raise ValueError(f"trait_column_name={trait_column_name} is invalid.")
            ax.set_ylabel(r"absolute order price scale $|\hat{r}_{t_i^j}^j|$")
        for decision_histories_df in decision_histories_dfs:
            for ax, trait_column_name in zip(axes, trait_column_names):
                trait_value_arr: ndarray = decision_histories_df[trait_column_name].values
                order_price_scale_arr: ndarray = decision_histories_df["order_price_scale"].values
                abs_order_price_scale_arr = np.abs(order_price_scale_arr)
                ax.scatter(trait_value_arr, abs_order_price_scale_arr, s=1, c="black")
        for fig, save_name in zip(figs, save_names):
            save_path: Path = self.figs_save_path / save_name
            fig.savefig(save_path, bbox_inches="tight")

    def _get_trait_pl(
        self,
        decision_histories_df: DataFrame
    ) -> tuple[dict[str, list[float]], list[float]]:
        """Get trait values and profit or loss."""
        trait_dic: dict[str, list[float]] = {
            "skill_boundedness": [], "risk_aversion_term": [], "discount_factor": []
        }
        pls: list[float] = []
        agent_ids: list[int] = list(decision_histories_df["agent_id"].unique())
        for agent_id in agent_ids:
            agent_df: DataFrame = decision_histories_df[
                decision_histories_df["agent_id"] == agent_id
            ]
            asset_volume_arr: ndarray = agent_df["asset_volume"].values
            cash_amount_arr: ndarray = agent_df["cash_amount"].values
            market_prices: ndarray = agent_df["market_price"].values
            wealth_arr: ndarray = asset_volume_arr * market_prices + cash_amount_arr
            pl: float = np.log(wealth_arr[-1]) - np.log(wealth_arr[0])
            trait_dic["skill_boundedness"].append(agent_df["skill_boundedness"].values[0])
            trait_dic["risk_aversion_term"].append(agent_df["risk_aversion_term"].values[0])
            trait_dic["discount_factor"].append(agent_df["discount_factor"].values[0])
            pls.append(pl)
        return trait_dic, pls
    
    def draw_actions_given_obs2d(
        self,
        target_obs_names: list[str],
        target_obs_indices: list[int],
        target_action_idx: int,
        x_obs_values: list[float],
        y_obs_values: list[float],
        initial_obs_values: list[float],
        save_name: str,
    ) -> None:
        assert len(target_obs_names) == 2
        assert len(target_obs_indices) == 2
        x_obs_name, y_obs_name = target_obs_names
        x_obs_idx, y_obs_idx = target_obs_indices
        obses_arr: ndarray = np.tile(
            np.array(initial_obs_values),
            (len(y_obs_values), len(x_obs_values), 1)
        )
        obses_arr[:,:,y_obs_idx] = np.tile(y_obs_values, (len(x_obs_values),1)).T
        obses_arr[:,:,x_obs_idx] = np.tile(x_obs_values, (len(y_obs_values),1))
        actions_arr: ndarray = self.algo.exploit(obses_arr)
        target_action_arr: ndarray = actions_arr[:,:,target_action_idx]
        fig: Figure = plt.figure(figsize=(len(y_obs_values),2*len(x_obs_values)))
        # ax: Axes = fig.add_subplot(1,1,1)
        # heatmap = ax.pcolor(target_action_arr)
        # ax.set_xticks(np.arange(len(x_obs_values))+0.5)
        # ax.set_yticks(np.arange(len(y_obs_values))+0.5)
        # ax.set_xticklabels(
        #     [f"{v:.2f}" for v in x_obs_values],
        #     rotation=45
        # )
        # ax.set_yticklabels([f"{v:.3f}" for v in y_obs_values])
        # ax.set_xlabel(x_obs_name)
        # ax.set_ylabel(y_obs_name)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(heatmap, ax=ax, cax=cax)
        # ax.set_aspect("equal", adjustable="box")
        ax: Axes = sns.heatmap(
            target_action_arr, cmap="OrRd", cbar=True,
            annot=True, fmt=".2f", annot_kws={"fontsize": 8}
        )
        ax.set_xticklabels(
            [f"{v:.2f}" for v in x_obs_values],
            rotation=45
        )
        ax.set_yticklabels([f"{v:.3f}" for v in y_obs_values])
        ax.set_xlabel(x_obs_name)
        ax.set_ylabel(y_obs_name)
        save_path: Path = self.figs_save_path / save_name
        fig.savefig(save_path, bbox_inches="tight")

    def draw_actions_given_obs1d(
        self,
        target_obs_name: str,
        target_obs_idx: int,
        target_action_name: str,
        target_action_idx: int,
        obs_values: list[float],
        initial_obs_values: list[float],
        save_name: str,
    ) -> None:
        obses_arr: ndarray = np.tile(
            np.array(initial_obs_values),
            (len(obs_values), 1)
        )
        obses_arr[:,target_obs_idx] = np.array(obs_values)
        actions_arr: ndarray = self.algo.exploit(obses_arr)
        target_action_arr: ndarray = actions_arr[:,target_action_idx]
        fig: Figure = plt.figure(figsize=(15, 8))
        # ax: Axes = fig.add_subplot(1,1,1)
        # ax.plot(obs_values, target_action_arr, c="black")
        # ax.set_xlabel(target_obs_name)
        # ax.set_ylabel(target_action_name)
        ax: Axes = sns.lineplot(
            x=obs_values, y=target_action_arr
        )
        ax.set_xlabel(target_obs_name)
        ax.set_ylabel(target_action_name)
        save_path: Path = self.figs_save_path / save_name
        fig.savefig(save_path, bbox_inches="tight")
