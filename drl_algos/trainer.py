from pettingzoo import AECEnv
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
import torch
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from algorithm import Algorithm
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
        self.train_env.seed(seed)
        self.test_env: AECEnv = test_env
        self.test_env.seed(seed+42)
        self.algo: Algorithm = algo
        self.actor_best_save_path: Optional[Path] = actor_best_save_path
        self.actor_last_save_path: Optional[Path] = actor_last_save_path
        self.results_dic: dict[str, dict[AgentID, list[float]]] = self._set_results_dic(other_indicators)
        self.num_train_steps: int = int(num_train_steps)
        self.eval_interval: int = int(eval_interval)
        self.num_eval_episodes: int = int(num_eval_episodes)
        self.best_reward: float = -1e+10

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
        results_dic: dict[str, list[int] | dict[AgentID, list[float]]] = {}
        results_dic["step"] = []
        results_dic["total_reward"] = {
            agent_id: [] for agent_id in self.train_env.agents
        }
        for indicator in other_indicators:
            results_dic[indicator] = {
                agent_id: [] for agent_id in self.train_env.agents
            }
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
        total_reward_dic: dict[AgentID, list[float]] = {
            agent_id: [] for agent_id in self.test_env.agents
        }
        for _ in range(self.num_eval_episodes):
            self.test_env.reset()
            done: bool = False
            episode_reward_dic: dict[AgentID, float] = {
                agent_id: 0.0 for agent_id in self.test_env.agents
            }
            while not done:
                for agent_id in self.test_env.agent_iter():
                    obs: ObsType = self.test_env.observe(agent_id)
                    action: ActionType = self.algo.exploit(obs)
                    reward, done, info = self.test_env.step(action)
                    episode_reward_dic[agent_id] += reward
            for agent_id in self.test_env.agents:
                total_reward_dic[agent_id].append(episode_reward_dic[agent_id])
        average_total_reward_dic: dict[AgentID, float] = {
            agent_id: np.mean(total_reward_dic[agent_id]) for agent_id in self.test_env.agents
        }
        self._record_indicators(
            current_total_steps, average_total_reward_dic, info
        )
        self._save_actor(current_total_steps, average_total_reward_dic)

    def _record_indicators(
        self,
        current_total_steps: int,
        average_total_reward_dic: dict[AgentID, float],
        info: dict,
    ) -> None:
        """Record indicators.
        
        Record indicators in self.results_dic.

        Args:
            current_total_steps (int): Current total steps.
            average_total_reward_dic (dict[AgentID, float]): Average total rewards.
            info (Any): Other indicators.
        """
        self.results_dic["step"].append(current_total_steps)
        for agent_id, average_total_reward in average_total_reward_dic.items():
            self.results_dic["total_reward"][agent_id].append(average_total_reward)
        for key, value in info.items():
            if key in self.results_dic:
                self.results_dic[key].append(value)

    def _save_actor(
        self,
        average_total_reward_dic: dict[AgentID, float],
    ) -> None:
        eval_average_reward: float = np.mean(average_total_reward_dic.values())
        if eval_average_reward < self.best_reward:
            if self.actor_last_save_path is None:
                return
            save_path = self.actor_last_save_path
        else:
            if self.actor_best_save_path is None:
                return
            self.best_reward = eval_average_reward
            save_path = self.actor_best_save_path
        torch.save(
            {
                "actor_state_dict": self.algo.actor.state_dict(),
                "results_dic": self.results_dic
            },
            str(save_path)
        )
        print(f"model saved to >> {str(save_path)}")
