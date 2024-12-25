from abc import abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class RolloutBuffer4IPPO:
    """Rollout buffer for IPPO class.
    
    Rollout buffer is usually used to store experiences while training on-policy RL algorithm.
    Since RolloutBufferForMAPPO is for independent PPO algorithm, the buffer store experiences of all agents.

    An experience consists of (obs, action, reward, next_obs, done, log_prob).
    """
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        device: torch.device
    ) -> None:
        """Initialize RolloutBufferForMAPPO.

        Args:
            buffer_size (int): Buffer size.
            obs_shape (tuple[int]): Observation shape.
            action_shape (tuple[int]): Action shape.
            num_agents (int): Number of agents.
            device (torch.device): Device.
        """
        self.buffer_size: int = int(buffer_size)
        self.num_agents: int = int(num_agents)
        self.obs_shape: tuple[int] = obs_shape
        self.action_shape: tuple[int] = action_shape
        self.device: torch.device = device
        self.initialize_buffer()

    def initialize_buffer(self) -> None:
        """Initialize buffer.
        
        RolloutBuffer4IPPO stores rollout experiences of all agents.
        """
        self.is_storing_dic: dict[int, bool] = {
            agent_idx: True for agent_idx in range(self.num_agents)
        }
        self.next_idx_dic: dict[int, int] = {
            agent_idx: 0 for agent_idx in range(self.num_agents)
        }
        self.obses: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, *self.obs_shape),
            dtype=torch.float, device=self.device
        )
        self.actions: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, *self.action_shape),
            dtype=torch.float, device=self.device
        )
        self.rewards: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, 1),
            dtype=torch.float, device=self.device
        )
        self.rewards: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, 1),
            dtype=torch.float, device=self.device
        )
        self.dones: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, 1),
            dtype=torch.float, device=self.device
        )
        self.log_probs: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, 1),
            dtype=torch.float, device=self.device
        )
        self.next_obses: Tensor = torch.empty(
            (self.num_agents, self.buffer_size, *self.obs_shape),
            dtype=torch.float, device=self.device
        )

    def append(
        self,
        agent_idx: int,
        obs_tensor: Tensor,
        action_tensor: Tensor,
        reward: float,
        done: bool,
        log_prob: float
    ) -> None:
        """add one experience to buffer.
        
        RolloutBuffer4IPPO synchronously stores experiences of all agents. In other words,
        .append() method will not append filled agents' experiences to the buffer until all buffer will be filled.
        
        """
        next_idx: int = self.next_idx_dic[agent_idx]
        is_storing: bool = self.is_storing_dic[agent_idx]
        if not is_storing:
            if next_idx == 0:
                self.obses[agent_idx, -1].copy_(
                    obs_tensor.view(self.obs_shape)
                )
            self.next_idx_dic[agent_idx] = 1
            return
        else:
            self.next_obses[agent_idx, next_idx-1].copy_(
                obs_tensor.view(self.obs_shape)
            )
        self.obses[agent_idx, next_idx].copy_(obs_tensor.view(self.obs_shape))
        self.actions[agent_idx, next_idx].copy_(action_tensor.view(self.action_shape))
        self.rewards[agent_idx, next_idx] = float(reward)
        self.dones[agent_idx, next_idx] = float(done)
        self.log_probs[agent_idx, next_idx] = float(log_prob)
        self.next_idx_dic[agent_idx] = (next_idx + 1) % self.buffer_size
        if next_idx + 1 == self.buffer_size:
            self.is_storing_dic[agent_idx] = False

    def is_filled(self) -> bool:
        """Check if the buffer is filled.
        
        Returns:
            bool: If the buffer is filled.
        """
        return not all(self.is_storing_dic.values())

    def get(self) -> tuple[Tensor]:
        """Get all experiences.
        
        Returns:
            experiences (tuple[Tensor]): All experiences.
        """
        filled_indices: ndarray = np.where(
            np.array(list(self.is_storing_dic.values())) == False
        )[0]
        normed_rewards: Tensor = self.rewards[filled_indices] / (self.rewards[filled_indices].std() + 1e-06)
        #print(normed_rewards.quantile(0.01), normed_rewards.quantile(0.25), normed_rewards.quantile(0.5), normed_rewards.quantile(0.75), normed_rewards.quantile(0.99))
        normed_rewards = normed_rewards.clamp(-5, 5)
        experiences: tuple[Tensor] = (
            self.obses[filled_indices],
            self.actions[filled_indices],
            normed_rewards,
            self.dones[filled_indices],
            self.log_probs[filled_indices],
            self.next_obses[filled_indices]
        )
        for agent_idx in filled_indices:
            self.is_storing_dic[agent_idx] = True
            self.next_idx_dic[agent_idx] = 0
        return experiences
