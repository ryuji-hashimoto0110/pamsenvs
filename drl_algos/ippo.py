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
from buffers import RolloutBuffer4IPPO
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

class IPPOActor(Module):
    """IPPO actor class.

    Approximate Individual Proximal Policy Optimization (IPPO) policy.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        action_shape: ndarray,
        device: torch.device
    ) -> None:
        """Initialize IPPOActor.

        Args:
            obs_shape (ndarray): Observation shape of each agent. Assume that all agents have the same observation shape.
            action_shape (ndarray): Action shape of each agent. Assume that all agents have the same action shape.
            device (torch.device): Device.
        """
        super(IPPOActor, self).__init__()
        self.obs_shape: ndarray = obs_shape
        self.actlayer: Module = nn.Sequential(
            nn.Linear(np.prod(obs_shape), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, np.prod(action_shape)),
        ).to(device)
        initialize_module_orthogonal(self.actlayer)
        self.log_stds: Tensor = nn.Parameter(
            torch.zeros(1, action_shape[0])
        ).to(device)

    def _resize_obses(self, obses: Tensor) -> Tensor:
        """Resize observation tensor."""
        if obses.shape == self.obs_shape:
            obses = obses.unsqueeze_(0)
        elif obses[0].shape != self.obs_shape:
            raise ValueError(f"Invalid observation shape: {obses.shape}")
        return obses

    def forward(self, obses: Tensor) -> Tensor:
        """calc deterministic action."""
        obses = self._resize_obses(obses)
        return torch.tanh(self.actlayer(obses)).clamp(-0.999, 0.999)
    
    def sample(self, obses: Tensor) -> tuple[Tensor]:
        """Sample action and calculate log probability."""
        obses = self._resize_obses(obses)
        means: Tensor = self.actlayer(obses)
        actions, log_prob = reparametrize(
            means, self.log_stds.clamp(-20,2)
        )
        actions = actions.clamp(-0.999, 0.999)
        return actions, log_prob
    
    def calc_log_prob(
        self,
        obses: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Calculate log probability."""
        obses = self._resize_obses(obses)
        means: Tensor = self.actlayer(obses)
        if actions.shape != means.shape:
            raise ValueError(f"Invalid action shape: {actions.shape}.")
        noises: Tensor = (actions - means) / (self.log_stds.exp() + 1e-06)
        log_probs: Tensor = calc_log_prob(self.log_stds, noises, actions)
        return log_probs


class IPPOCritic(Module):
    """IPPO critic class.
    
    Estimate the value of the observation.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        device: torch.device
    ) -> None:
        """Initialize IPPOCritic.

        Args:
            obs_shape (ndarray): Observation shape of each agent. Assume that all agents have the same observation shape.
            device (torch.device): Device.
        """
        super(IPPOCritic, self).__init__()
        self.obs_shape: ndarray = obs_shape
        self.valuelayer: Module = nn.Sequential(
            nn.Linear(np.prod(obs_shape), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        ).to(device)
        initialize_module_orthogonal(self.valuelayer)

    def _resize_obses(self, obses: Tensor) -> Tensor:
        """Resize observation tensor."""
        if obses.shape == self.obs_shape:
            obses = obses.unsqueeze_(0)
        elif obses[0].shape != self.obs_shape:
            raise ValueError(f"Invalid observation shape: {obses.shape}")
        return obses
    
    def forward(self, obses: Tensor) -> Tensor:
        """Forward method."""
        obses = self._resize_obses(obses)
        return self.valuelayer(obses)


class IPPO(Algorithm):
    """IPPO algorithm class."""
    def __init__(
        self,
        device: str,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        num_agents: int,
        seed: int = 42,
        rollout_length: int = 512,
        num_updates_per_rollout: int = 1,
        batch_size: int = 512,
        gamma: float = 0.99,
        gamma_idx: Optional[int] = None,
        lr_actor: float = 3e-04,
        lr_critic: float = 3e-04,
        clip_eps: float = 0.2,
        lmd: float = 0.95,
        max_grad_norm: float = 0.5,
        display_process: bool = True
    ) -> None:
        """initialization.
        
        Args:
            obs_shape (tuple[int]): Observation shape of the environment.
            action_shape (tuple[int]): Action shape of the environment.
            seed (int): Random seed.
            rollout_length (int): Length of a rollout (a sequence of experiences)
                that can be stored in the buffer. (=buffer_size). Defaults to 2048.
            num_updates_per_rollout (int): The number of times to update the network
                using one rollout. Defaults to 10.
            batch_size (int): Batch size. A rollout is processed in mini batch. Defaults to 1024
            gamma (float): Discount factor. Defaults to 0.995.
            gamma_idx (int, optional): Index of gamma in the observation. If gamma is included in the observation, set gamma_idx.
                Defaults to None.
            lr_actor (float): Learning rate for the actor. Defaults to 3e-04.
            lr_critic (float): Learning rate for the critic. Defaults to 3e-04.
            clip_eps (float): The value to clip importance_ratio (pi / pi_old)
                used in Clipped Surrogate Objective. clip_eps is also used in Value Clipping.
                Defaults to 0.2.
            lmd (float): The value to determine how much future TD errors are important
                in Generalized Advantage Estimation (GAE) . Defaults to 0.97.
            max_grad_norm (float): Threshold to clip the norm of the gradient.
                Gradient clipping is used to avoid exploding gradients. Defaults to 0.5.
        """
        super(IPPO, self).__init__(device=device)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        self.obs_shape: tuple[int] = obs_shape
        self.action_shape: tuple[int] = action_shape
        self.buffer: RolloutBuffer4IPPO = RolloutBuffer4IPPO(
            buffer_size=rollout_length, num_agents=num_agents,
            obs_shape=obs_shape, action_shape=action_shape,
            device=self.device
        )
        self.actor: IPPOActor = IPPOActor(obs_shape, action_shape, self.device)
        self.critic: IPPOCritic = IPPOCritic(obs_shape, self.device)
        self.optim_actor: Optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        #self.scheduler_actor = optim.lr_scheduler.LambdaLR(
        #    self.optim_actor,
        #    lr_lambda=lambda epoch: max(1e-06 / lr_actor, 0.995**(epoch//50))
        #)
        self.optim_critic: Optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        #self.scheduler_critic = optim.lr_scheduler.LambdaLR(
        #    self.optim_critic,
        #    lr_lambda=lambda epoch: max(2e-06 / lr_critic, 0.995**(epoch//50))
        #)
        self.rollout_length: int = rollout_length
        self.num_updates_per_rollout: int = num_updates_per_rollout
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.gamma_idx: Optional[int] = gamma_idx
        self.clip_eps: float = clip_eps
        self.lmd: float = lmd
        self.max_grad_norm: float = max_grad_norm
        self.agent_id2agent_idx_dic: dict[AgentID, int] = {}
        if display_process:
            print("[bold green]IPPO[/bold green]")
            print(f"device: {self.device} obs: {obs_shape} action: {action_shape}")
            print(f"num agents: {num_agents} buffer size: {rollout_length}")
            print(f"num updates per rollout: {num_updates_per_rollout} batch size: {batch_size}")
            print(f"gamma: {gamma} gamma idx: {gamma_idx} lr actor: {lr_actor} lr critic: {lr_critic}")
            print(f"clip epsilon: {clip_eps} lambda: {lmd} max grad norm: {max_grad_norm}")
            print()

    def assign_agent_id2agent_idx(self, agent_ids: list[AgentID]) -> None:
        """Assign agent_id to agent_idx. Usually called by Trainer."""
        for idx, id in enumerate(sorted(agent_ids)):
            self.agent_id2agent_idx_dic[id] = idx

    def is_ready_to_update(self, current_total_steps):
        return self.buffer.is_filled()
    
    def _initialize_buffer(self):
        self.buffer.initialize_buffer()

    def _store_experience(
        self,
        agent_id: AgentID,
        obs_tensor: Tensor,
        action_tensor: Tensor,
        reward: float,
        done: bool,
        log_prob: float
    ) -> None:
        """Store experience in the buffer."""
        agent_idx: int = self.agent_id2agent_idx_dic[agent_id]
        self.buffer.append(
            agent_idx=agent_idx, obs_tensor=obs_tensor,
            action_tensor=action_tensor, reward=reward,
            done=done, log_prob=log_prob
        )

    def update(self):
        """update actor and critic.

        get the rollout from self.buffer and update actor and critic using the rollout.
        target obs value R(lmd) and GAE is calculated in advance.
        The rollout is divided into mini-batches and processes in sequence.

        obses_all (num_agents, buffer_size, obs_shape)
        obses (buffer_size, obs_shape)
        """
        obses_all, actions_all, rewards_all, dones_all, log_probs_old_all, next_obses_all = \
            self.buffer.get()
        num_agents, buffer_size, _ = obses_all.shape
        for obses, actions, rewards, dones, log_probs_old, next_obses in zip(
            obses_all, actions_all, rewards_all, dones_all, log_probs_old_all, next_obses_all
        ):
            assert buffer_size == len(obses)
            with torch.no_grad():
                values: Tensor = self.critic(obses)
                next_values: Tensor = self.critic(next_obses)
            if self.gamma_idx is not None:
                gamma: Tensor = obses[:, self.gamma_idx].unsqueeze_(1)
                targets, advantages = self.calc_gae(
                    values, rewards, dones, next_values, gamma, self.lmd
                )
            else:
                targets, advantages = self.calc_gae(
                    values, rewards, dones, next_values, self.gamma, self.lmd
                )
            for _ in range(self.num_updates_per_rollout):
                indices: np.ndarray = np.arange(self.rollout_length)
                np.random.shuffle(indices)
                for start in range(0, self.rollout_length, self.batch_size):
                    sub_indices = indices[start:start+self.batch_size]
                    self.update_critic(obses[sub_indices], targets[sub_indices])
                    self.update_actor(
                        obses[sub_indices],
                        actions[sub_indices],
                        log_probs_old[sub_indices],
                        advantages[sub_indices]
                    )
            #self.scheduler_actor.step()
            #self.scheduler_critic.step()
            #print(self.scheduler_actor.get_lr())

    def calc_gae(
        self,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_values: Tensor,
        gamma: float = 0.995,
        lmd: float = 0.99
    ) -> tuple[Tensor]:
        """calculate target obs value R(lmd) and Generalized Advantage Estimation (GAE).

        The objective function in PPO contains Advantage function.
        According to the PPO paper, GAE is used as Advantage function.
        GAE is calculated recursivelly using TD(1) error.
        GAE is scaled. It is known that reward scaling has a strong learning stabilizing effect in continuous-action settings.
        Target obs value R(lmd) is calculated as weighted average of TD(lam) error + V(s).
        """
        td_err: Tensor = rewards + gamma * next_values * (1 - dones) - values
        advantages: Tensor = torch.empty_like(rewards)
        advantages[-1] = td_err[-1]
        for t in reversed(range(len(rewards)-1)):
            advantages[t] = td_err[t] + gamma[t] * lmd * (1-dones[t]) * advantages[t+1]
        targets: Tensor = advantages + values
        advantages = advantages / (advantages.std() + 1e-08)
        return targets, advantages

    def update_critic(
        self,
        obses: Tensor,
        targets: Tensor
    ) -> None:
        """update critic using experiences.
        """
        loss_critic: Tensor = (self.critic(obses) - targets).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(
        self,
        obses: Tensor,
        actions: Tensor,
        log_probs_old: Tensor,
        advantages: Tensor
    ) -> None:
        """update actor using experiences.
        """
        log_probs_now: Tensor = self.actor.calc_log_prob(obses, actions)
        importance_ratio = (log_probs_now - log_probs_old).exp()
        loss_actor1: Tensor = - importance_ratio * advantages
        loss_actor2: Tensor = - torch.clamp(
            importance_ratio,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor: Tensor = torch.max(loss_actor1, loss_actor2).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

