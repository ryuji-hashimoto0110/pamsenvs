from pettingzoo import AECEnv
import numpy as np
from numpy import ndarray
from pams.market import Market
import pathlib
from pathlib import Path
import torch
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from algorithm import Algorithm
from rich import print
from typing import Any
from typing import Optional
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class Trainer:
    """Trainer class.
    
    train and evaluate RL algorithm.
    """
    def __init__(
        self,
        train_env: AECEnv,
        test_env: AECEnv,
        algo: Algorithm,
        seed: int = 42,
        actor_best_save_path: Optional[Path] = None,
        actor_last_save_path: Optional[Path] = None,
        other_indicators: list[str] = [],
        num_train_steps: int = 1e+07,
        eval_interval: int = 1e+05,
        num_eval_episodes: int = 10,
        display_process: bool = True
    ) -> None:
        """initialization.
        
        Args:
            train_env (AECEnv): Training environment.
            test_env (AECEnv): Test environment.
            algo (Algorithm): RL algorithm.
            seed (int): Random seed. Default to 42.
            actor_best_save_path (Path, optional): path to save actor's weights with best eval rewards. Default to None.
            actor_last_save_path (Path. optional): path to save actor's weights with last eval rewards. Default to None.
            other_indicators (list[str]): Other indicators to be saved. Default to [].
            num_train_steps (int): Number of training steps. Default to 10,000,000.
            eval_interval (int): Evaluation interval. Default to 100,000.
            num_eval_episodes (int): Number of evaluation episodes. Default to 10.
        """
        self.train_env: AECEnv = train_env
        self.seed: int = seed
        self.train_env.seed(self.seed)
        self.test_env: AECEnv = test_env
        self.start_exploit_time: int = self._get_start_exploit_time(self.test_env.config_dic)
        self.algo: Algorithm = algo
        self.actor_best_save_path: Optional[Path] = actor_best_save_path
        self.actor_last_save_path: Optional[Path] = actor_last_save_path
        self.results_dic: dict[str, list[float]] = self._set_results_dic(other_indicators)
        self.num_train_steps: int = int(num_train_steps)
        self.eval_interval: int = int(eval_interval)
        self.num_eval_episodes: int = int(num_eval_episodes)
        self.best_reward: float = -1e+10
        self.display_process: bool = display_process
        if self.display_process:
            print("[bold green]Trainer[/bold green]")
            print(f"env: {train_env}")
            print(f"train steps: {num_train_steps} eval interval: {eval_interval}")
            print(f"num eval episodes: {num_eval_episodes}")

    def _get_start_exploit_time(self, config: dict[str, Any]) -> int:
        return config["simulation"]["sessions"][0]["iterationSteps"]

    def _set_results_dic(
        self, other_indicators: list[str]
    ) -> dict[str, list[int] | dict[AgentID, list[float]]]:
        """Set results dictionary.
        
        results_dic = {
            "step": [],
            "total_reward: {
                AgentID: []
            }
        }

        Args:
            other_indicators (list[str]): Other indicators to be saved.
        
        Returns:
            results_dic (dict[str, dict[AgentID, list[float]]]): Results dictionary.
        """
        results_dic: dict[str, list[int] | list[float]] = {}
        results_dic["step"] = []
        results_dic["total_reward"] = []
        for indicator in other_indicators:
            results_dic[indicator] = []
        return results_dic
    
    def train(self) -> None:
        """train the algorithm.
        
        Train self.algo for num_train_steps steps in self.train_env environment.
        Evaluate the algorithm every eval_interval steps in self.test_env environment.
        """
        current_episode_steps: int = 0
        self.train_env.reset()
        if hasattr(self.algo, "assign_agent_id2agent_idx"):
            self.algo.assign_agent_id2agent_idx(self.train_env.agents)
        for current_total_steps in range(1, self.num_train_steps):
            current_episode_steps = self.algo.step(
                env=self.train_env,
                current_episode_steps=current_episode_steps,
                current_total_steps=current_total_steps,
            )
            if self.algo.is_ready_to_update(current_total_steps):
                self.algo.update()
            if current_total_steps % self.eval_interval == 0:
                self.evaluate(current_total_steps)

    def evaluate(self, current_total_steps: int) -> None:
        """evaluate algorithm.

        Run episodes self.num_eval_episodes times in self.test_env environment.
        Record average cumulative rewards and other indicators.
        Save paremeters of the trained network.

        Args:
            current_total_steps (int): Current total steps.
        """
        average_total_reward: float = 0.0
        average_total_execution_volume: float = 0.0
        average_price_range: float = 0.0
        obs_dics: list[dict[int, list[float | int]]] = []
        action_dics: list[dict[int, list[float | int]]] = []
        reward_dics: list[dict[int, list[float | int]]] = []
        self.test_env.seed(self.seed+42)
        for _ in range(self.num_eval_episodes):
            total_execution_volume: int = 0
            self.test_env.reset()
            done: bool = False
            episode_reward_dic: dict[AgentID, float] = {
                agent_id: 0.0 for agent_id in self.test_env.agents
            }
            episode_length: int = 0
            for agent_id in self.test_env.agent_iter():
                obs: ObsType = self.test_env.last()
                action: ActionType = self.algo.exploit(obs)
                reward, done, info = self.test_env.step(action)
                episode_reward_dic[agent_id] += reward
                if "execution_volume" in info:
                    total_execution_volume += info["execution_volume"]
                if done:
                    break
                else:
                    episode_length += 1
            average_total_reward += np.sum(
                list(episode_reward_dic.values())
            ) / self.num_eval_episodes
            average_total_execution_volume += total_execution_volume / self.num_eval_episodes
            average_price_range += self._get_price_range(self.test_env) / self.num_eval_episodes
            if hasattr(self.test_env, "obs_dic"):
                obs_dics.append(self.test_env.obs_dic)
            if hasattr(self.test_env, "action_dic"):
                action_dics.append(self.test_env.action_dic)
            if hasattr(self.test_env, "reward_dic"):
                reward_dics.append(self.test_env.reward_dic)
        self.results_dic["step"].append(current_total_steps)
        self.results_dic["total_reward"].append(average_total_reward)
        if "execution_volume" in self.results_dic:
            self.results_dic["execution_volume"].append(average_total_execution_volume)
        if "lr_actor" in self.results_dic:
            if hasattr(self.algo, "scheduler_actor"):
                self.results_dic["lr_actor"].append(self.algo.scheduler_actor.get_lr())
        if "price_range" in self.results_dic:
            self.results_dic["price_range"].append(average_price_range)
        if "obs_dics" in self.results_dic:
            self.results_dic["obs_dics"] = obs_dics
        if "action_dics" in self.results_dic:
            self.results_dic["action_dics"] = action_dics
        if "reward_dics" in self.results_dic:
            self.results_dic["reward_dics"] = reward_dics
        print(f"step: {current_total_steps}, total reward: {average_total_reward:.2f}, total execution volume: {average_total_execution_volume:.2f}, price range: {average_price_range:.2f}")
        self._save_actor(average_total_reward)

    def _get_price_range(self, env: AECEnv) -> float:
        price_range: float = 0.0
        if hasattr(env, "markets"):
            market: Market = env.markets[0]
            prices: list[float] = market.get_market_prices()
            price_range = np.max(prices) - np.min(prices)
        return price_range

    def _save_actor(self, average_total_reward: float) -> None:
        if average_total_reward < self.best_reward:
            if self.actor_last_save_path is None:
                return
            save_path = self.actor_last_save_path
        else:
            if self.actor_best_save_path is None:
                return
            self.best_reward = average_total_reward
            save_path = self.actor_best_save_path
        torch.save(
            {
                "actor_state_dict": self.algo.actor.state_dict(),
                "results_dic": self.results_dic
            },
            str(save_path)
        )
        print(f"model saved to >> {str(save_path)}")
        print()
