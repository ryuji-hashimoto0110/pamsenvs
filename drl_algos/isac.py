from abc import ABC
from abc import abstractmethod
import numpy as np
from numpy import ndarray
import pathlib
import pathlib
from pathlib import Path
import sys
curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
parent_path: pathlib.Path = curr_path.parents[0]
sys.path.append(str(parent_path))
from algorithm import Algorithm
from buffers import ReplayBuffer4MARL
from drl_utils import initialize_module_orthogonal
from drl_utils import calc_log_prob
from drl_utils import reparametrize
from rich import print
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import Optional
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class SACActor(Module):
    """SAC Actor class.

    Approximate Soft Actor Critic (SAC) policy.

    net(states) outputs the mean and variance of a diagonal gaussian distribution.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        action_shape: ndarray
    ) -> None:
        """initialization.

        Args:
            state_shape (np.ndarray): the shape of state. state_shape = state.shape
            action_shape (np.ndarray): the shape of action. action_shape = action.shape
        """
        super().__init__()
        self.obs_shape: ndarray = obs_shape
        self.actlayer: Module = nn.Sequential(
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2*np.prod(action_shape)),
        )
        initialize_module_orthogonal(self.actlayer)
    
    def _resize_obses(self, obses: Tensor) -> Tensor:
        """Resize observation tensor."""
        if obses.shape == self.obs_shape:
            obses = obses.unsqueeze_(0)
        return obses
    
    def forward(self, obses: Tensor) -> Tensor:
        """forward method.

        Return the mean with tanh applied.

        Args:
            states (torch.Tensor): (batch_size, *state_shape)
        Returns:
            actions (torch.Tensor): (batch_size, *action_shape)
        """
        obses = self._resize_obses(obses)
        means: Tensor
        means, _ = self.actlayer(obses).chunk(2, dim=-1)
        actions = torch.tanh(means).clamp(-0.999, 0.999)
        return actions

    def sample(self, obses: Tensor) -> tuple[Tensor]:
        """sample method.

        Sample from the distribution (diagonal gaussian + tanh) using reparametrization trick
        and calculate the result's probability density.

        Args:
            states (torch.Tensor): (batch_size, *state_shape)
        Returns:
            actions (torch.Tensor): (batch_size, *action_shape)
            log_pis (torch.Tensor): (batch_size, 1)
        """
        obses = self._resize_obses(obses)
        means: Tensor
        log_stds: Tensor
        means, log_stds = self.actlayer(obses).chunk(2, dim=-1)
        actions, log_probs = reparametrize(means, log_stds.clamp(-20, 2))
        actions = actions.clamp(-0.999, 0.999)
        return actions, log_probs

class SACCritic(Module):
    """SAC Critic class.

    Estimate the value of (state, action) pair (=Q function) returned by Soft Actor Critic (SAC) Actor.
    SACCritic consists of two indespensible network to estimate soft-state-action-value (Clipped Double Q, ref TD3).
    """
    def __init__(
        self,
        obs_shape: ndarray,
        action_shape: ndarray
    ) -> None:
        """initialization.

        Args:
            state_shape (np.ndarray): the shape of state. state_shape = state.shape
            action_shape (np.ndarray): the shape of action. action_shape = action.shape
        """
        super().__init__()
        self.net1: Module = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.net2: nn.Module = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        obses: Tensor,
        actions: Tensor
    ) -> tuple[Tensor]:
        """forward method.

        Args:
            states (torch.Tensor): (batch_size, *state_shape)
            actions (torch.Tensor): (batch_size, *action_shape)
        Returns:
            q1s (torch.Tensor): first candidates of Q values. (batch_size, 1)
            q2s (torch.Tensor): second candidates of Q values. (batch_size, 1)
        """
        x = torch.cat([obses, actions], dim=-1)
        q1s = self.net1(x)
        q2s = self.net2(x)
        return q1s, q2s

class ISAC(Algorithm):
    """ISAC algorithm class."""
    def __init__(
        self,
        device: str,
        obs_shape: ndarray,
        action_shape: ndarray,
        num_agents: int,
        seed: int = 42,
        gamma: float = 0.99,
        gamma_idx: Optional[int] = None,
        gamma_min: float = 0.90,
        gamma_max: float = 0.999,
        lr_actor: float = 3e-04,
        lr_critic: float = 3e-04,
        batch_size: int = 1024,
        buffer_size: int = 2048,
        start_steps: int = 8000,
        tau: float = 5e-03,
        alpha: float = 0.20,
    ) -> None:
        super(ISAC, self).__init__(
            device=device,
            gamma=gamma,
            gamma_idx=gamma_idx,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        self.batch_size: int = batch_size
        self.buffer: ReplayBuffer4MARL = ReplayBuffer4MARL(
            buffer_size=buffer_size, num_agents=num_agents,
            obs_shape=obs_shape, action_shape=action_shape,
            device=device
        )
        self.actor: SACActor = SACActor(obs_shape, action_shape).to(device)
        self.critic: SACCritic = SACCritic(obs_shape, action_shape).to(device)
        self.target_critic: SACCritic = SACCritic(obs_shape, action_shape).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.optim_actor: Optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic: Optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau: float = tau
        self.alpha: float = alpha
        self.start_steps: int = start_steps

    def _initialize_buffer(self):
        pass

    def is_ready_to_update(self, current_total_steps):
        return current_total_steps >= max(self.start_steps, self.batch_size)
    
    def _store_experience(
        self,
        agent_id: AgentID,
        obs_tensor: Tensor,
        action_tensor: Tensor,
        reward: float,
        done: bool,
        log_prob: float = 0.00,
    ) -> None:
        agent_idx: int = self.agent_id2agent_idx_dic[agent_id]
        self.buffer.append(
            agent_idx=agent_idx,
            obs_tensor=obs_tensor,
            action_tensor=action_tensor,
            reward=reward,
            done=done,
            log_prob=log_prob,
        )

    def update(self) -> None:
        obses, actions, rewards, dones, next_obses, = self.buffer.sample(self.batch_size)
        self.update_critic(obses, actions, rewards, dones, next_obses)
        self.update_actor(obses)
        self.update_target_critic()

    def update_critic(
        self,
        obses: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_obses: Tensor,
    ) -> None:
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obses)
            next_q1s, next_q2s = self.target_critic(next_obses, next_actions)
            vs = torch.min(next_q1s, next_q2s) - self.alpha * next_log_probs
        if self.gamma_idx is not None:
            gamma: Tensor = obses[:, self.gamma_idx].unsqueeze_(1)
            gamma = self._re_preprocess_gamma(gamma)
        else:
            gamma = self.gamma
        qs_ = rewards + gamma * (1 - dones) * vs
        q1s, q2s = self.critic(obses, actions)
        loss_critic1: Tensor = (q1s - qs_).pow(2).mean()
        loss_critic2: Tensor = (q2s - qs_).pow(2).mean()
        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward()
        self.optim_critic.step()

    def update_actor(self, obses: Tensor) -> None:
        actions, log_probs = self.actor.sample(obses)
        q1s, q2s = self.critic(obses, actions)
        loss_actor: Tensor = (self.alpha * log_probs - torch.min(q1s, q2s)).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

    def update_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(param.data * self.tau)
