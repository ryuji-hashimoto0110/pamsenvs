from abc import ABC, abstractmethod
from numpy import ndarray
from pettingzoo import AECEnv
import torch
from torch import Tensor
from torch.nn import Module
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")

class Algorithm(ABC):
    """Algorithm base class.

    Most RL algorithms can be implemented as subclasses of this class.
    RL algorithm must contains
        self.actor (Module)
            .actor.forward(obs_tensor: Tensor) -> action_tensor: Tensor
            .actor.sample(obs_tensor: Tensor) -> action_tensor: Tensor, log_prob_tensor: float | Tensor
    and can contain
        self.critic (Module)
            .critic.forward(obs_tensor: Tensor, action_tensor: Tensor) -> value_tensor: Tensor
    """
    def __init__(self, device: str) -> None:
        self.actor: Module
        self.device: torch.device = torch.device(device)

    @torch.no_grad()
    def explore(
        self,
        obs: ObsType | Tensor,
    ) -> tuple[ActionType, float | Tensor]:
        """Explore the environment.

        Args:
            obs (ObsType | Tensor): Observation from the environment.

        Returns:
            action (ActionType): Action 
            log_prob (float | Tensor): log probability of the action.
        """
        obs_tensor: Tensor = self._convert_obs2tensor(obs)
        if not hasattr(self.actor, "sample"):
            raise ValueError("actoor.sample method is not implemented.")
        action_tensor, log_prob_tensor = self.actor.sample(obs_tensor)
        action: ActionType = self._convert_tensor2action(action_tensor)
        return action, log_prob_tensor
    
    @torch.no_grad()
    def exploit(
        self,
        obs: ObsType | Tensor,
    ) -> ActionType:
        """Exploit the environment.

        Args:
            obs (ObsType | Tensor): Observation from the environment.

        Returns:
            action (ActionType): Action.
        """
        obs_tensor: Tensor = self._convert_obs2tensor(obs)
        action_tensor = self.actor(obs_tensor)
        action: ActionType = self._convert_tensor2action(action_tensor)
        return action

    def _convert_obs2tensor(self, obs: ObsType | Tensor) -> Tensor:
        """Convert observation to tensor.

        This method may contain unsqueezing the observation tensor.

        Args:
            obs (ObsType | Tensor): Observation from the environment.

        Returns:
            obs_tensor (Tensor): Observation tensor.
        """
        return torch.tensor(obs, dtype=torch.float).unsqueeze_(0).to(self.device)
    
    def _convert_action2tensor(self, action: ActionType | Tensor) -> Tensor:
        return torch.tensor(action, dtype=torch.float).unsqueeze_(0).to(self.device)

    def _convert_tensor2action(self, action_tensor: Tensor | ActionType) -> ActionType:
        """Convert tensor to action.

        Args:
            action_tensor (Tensor | ActionType): Action tensor.

        Returns:
            action (ActionType): Action.
        """
        return action_tensor.detach().cpu().numpy()[0]
    
    @abstractmethod
    def is_ready_to_update(self, current_total_steps: int) -> bool:
        """Check if the algorithm is ready to update.

        For example, set true when the buffer is full.

        Args:
            current_total_steps (int): Current total steps.

        Returns:
            bool: If the algorithm is ready to update.
        """
        pass

    def step(
        self,
        env: AECEnv,
        current_episode_steps: int,
        current_total_steps: int,
    ) -> int:
        """Step the environment.

        Args:
            env (AECEnv): Environment.
            current_episode_steps (int): Current episode steps.
            current_total_steps (int): Current total steps.
        
        Returns:
            current_episode_steps (int): Updated current episode steps.
        """
        agent_id: AgentID = env.agent_selection
        obs: ObsType = env.last()
        action, log_prob = self.explore(obs)
        reward, done, _ = env.step(action)
        if done:
            env.reset()
        self._store_experience(
            agent_id=agent_id,
            obs_tensor=self._convert_obs2tensor(obs),
            action_tensor=self._convert_action2tensor(action),
            reward=reward,
            done=done,
            log_prob=log_prob,
        )
        if hasattr(env, "get_time"):
            current_episode_steps = env.get_time()
        else:
            current_episode_steps += 1
        return current_episode_steps

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def _store_experience(
        self,
        agent_id: AgentID,
        obs_tensor: Tensor,
        action_tensor: Tensor,
        reward: float,
        done: bool,
        log_prob: float
    ) -> None:
        pass

