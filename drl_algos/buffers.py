from abc import abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class ReplayBuffer4MARL:
    """Replay Buffer class.

    Replay buffer is usually used to store experiences while training off-policy RL algorithms.
    An experience consists of sets of five; (obs, action, reward, done, next_obs)
    """
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        device: torch.device
    ) -> None:
        """initialization.

        Args:
            buffer_size (int): maximum number of experiences that can be stored in the buffer.
            state_shape (np.ndarray): the shape of state. state_shape = state.shape
            action_shape (np.ndarray): the shape of action. action_shape = action.shape
            device (torch.device): Ex: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        """
        self.buffer_size: int = int(buffer_size)
        self.num_agents: int = int(num_agents)
        self.obs_shape: tuple[int] = obs_shape
        self.action_shape: tuple[int] = action_shape
        self.device: torch.device = device
        self.initialize_buffer()

    def initialize_buffer(self) -> None:
        """initialize buffer.
        """
        self.is_storing_dic: dict[int, bool] = {
            agent_idx: True for agent_idx in range(self.num_agents)
        }
        self.next_idx_dic: dict[int, int] = {
            agent_idx: 0 for agent_idx in range(self.num_agents)
        }
        self.experience_num_dic: dict[int, int] = {
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
        log_prob: float = 0.00
    ) -> None:
        """append experience to buffer.

        Experiences are collected as np.ndarray, so convert them to torch.Tensor.
        next_idx is the next index to store the experience in the buffer.
        If next_idx + 1 == buffer_size, that means the number of experiences (experience_num) reaches to buffer_size,
        then next_idx is set to be 0.
        """
        next_idx: int = self.next_idx_dic[agent_idx]
        is_storing: bool = self.is_storing_dic[agent_idx]
        if not is_storing:
            return
        else:
            self.rewards[agent_idx, next_idx-1] = float(reward)
            self.next_obses[agent_idx, next_idx-1].copy_(
                obs_tensor.view(self.obs_shape)
            )
            if next_idx == self.buffer_size:
                return
        self.obses[agent_idx, next_idx].copy_(obs_tensor.view(self.obs_shape))
        self.actions[agent_idx, next_idx].copy_(action_tensor.view(self.action_shape))
        self.dones[agent_idx, next_idx] = float(done)
        self.next_idx_dic[agent_idx] = (next_idx + 1) % self.buffer_size
        self.experience_num_dic[agent_idx] = min(
            self.experience_num_dic[agent_idx] + 1, self.buffer_size
        )

    def sample(self, batch_size: int) -> tuple[Tensor]:
        """sample experiences from buffer.

        Args:
            batch_size (int): the number of experiences to sample from buffer.

        Returns:
            experiences (tuple[Tensor]): sampled experiences.
        """
        total_experience_num: int = sum(self.experience_num_dic.values()) - len(self.experience_num_dic)
        obses: Tensor = torch.cat(
            [self.obses[agent_idx, :self.experience_num_dic[agent_idx]-1] for agent_idx in range(self.num_agents)],
            dim=0
        ) # (num_agents, buffer_size, obs_shape) -> (total_experience_num, obs_shape)
        assert obses.shape == (total_experience_num, *self.obs_shape)
        actions: Tensor = torch.cat(
            [self.actions[agent_idx, :self.experience_num_dic[agent_idx]-1] for agent_idx in range(self.num_agents)],
            dim=0
        )
        rewards: Tensor = torch.cat(
            [self.rewards[agent_idx, :self.experience_num_dic[agent_idx]-1] for agent_idx in range(self.num_agents)],
            dim=0
        )
        dones: Tensor = torch.cat(
            [self.dones[agent_idx, :self.experience_num_dic[agent_idx]-1] for agent_idx in range(self.num_agents)],
            dim=0
        )
        next_obses: Tensor = torch.cat(
            [self.next_obses[agent_idx, :self.experience_num_dic[agent_idx]-1] for agent_idx in range(self.num_agents)],
            dim=0
        )
        indices: ndarray = np.random.randint(0, total_experience_num, batch_size)
        experiences: tuple[Tensor] = (
            obses[indices],
            actions[indices],
            rewards[indices],
            dones[indices],
            next_obses[indices]
        )
        return experiences

class RolloutBuffer4IPPO(ReplayBuffer4MARL):
    """Rollout buffer for IPPO class.
    
    Rollout buffer is usually used to store experiences while training on-policy RL algorithm.
    Since RolloutBufferForMAPPO is for independent PPO algorithm, the buffer store experiences of all agents.

    An experience consists of (obs, action, reward, next_obs, done, log_prob).
    """
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
            return
        else:
            self.rewards[agent_idx, next_idx-1] = float(reward)
            self.next_obses[agent_idx, next_idx-1].copy_(
                obs_tensor.view(self.obs_shape)
            )
            if next_idx == self.buffer_size:
                self.is_storing_dic[agent_idx] = False
                return
        self.obses[agent_idx, next_idx].copy_(obs_tensor.view(self.obs_shape))
        self.actions[agent_idx, next_idx].copy_(action_tensor.view(self.action_shape))
        #self.rewards[agent_idx, next_idx] = float(reward)
        self.dones[agent_idx, next_idx] = float(done)
        self.log_probs[agent_idx, next_idx] = float(log_prob)
        self.next_idx_dic[agent_idx] = next_idx + 1

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
        #print(f"{normed_rewards.quantile(0.01):.4f}, {normed_rewards.quantile(0.05):.4f} {normed_rewards.quantile(0.95):.4f}, {normed_rewards.quantile(0.99):.4f}")
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
