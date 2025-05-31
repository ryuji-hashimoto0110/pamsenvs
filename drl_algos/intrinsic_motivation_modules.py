import numpy as np
from numpy import ndarray
import pathlib
import sys
curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
parent_path: pathlib.Path = curr_path.parents[0]
sys.path.append(str(parent_path))
from drl_utils import initialize_module_orthogonal
import torch
from torch.nn import Module
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
import torch.optim as optim


class RNDNetwork(Module):
    """Random Network Distillation (RND) Network.
    
    This network is used to compute intrinsic rewards based on the prediction error
    of a randomly initialized neural network.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        device: torch.device
    ) -> None:
        """Initialize IPPOCritic.

        Args:
            obs_shape (ndarray): Observation shape.
            device (torch.device): Device.
        """
        super(RNDNetwork, self).__init__()
        self.obs_shape: ndarray = obs_shape
        self.rndlayer: Module = nn.Sequential(
            nn.Linear(np.prod(obs_shape), 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        ).to(device)
        initialize_module_orthogonal(self.rndlayer, last_layer_scale=1.0)

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
        return self.rndlayer(obses)
    

class RNDRewardGenerator(Module):
    """Random Network Distillation (RND) Reward Generator.
    
    This module computes intrinsic rewards based on the prediction error of a randomly initialized neural network.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        device: torch.device
    ) -> None:
        """Initialize RNDRewardGenerator.

        Args:
            obs_shape (ndarray): Observation shape.
            device (torch.device): Device.
        """
        super(RNDRewardGenerator, self).__init__()
        self.obs_shape: ndarray = obs_shape
        self.rnd_predictor: RNDNetwork = RNDNetwork(obs_shape, device)
        self.rnd_predictor.train()
        self.rnd_target: RNDNetwork = RNDNetwork(obs_shape, device)
        self.rnd_target.eval()
        self.optimizer: Optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=3e-04)

    def generate_intrinsic_reward(
        self,
        obses: Tensor
    ) -> Tensor:
        """Generate intrinsic reward based on the prediction error of the RND network.

        Args:
            obses (Tensor): Observation tensor.

        Returns:
            Tensor: Intrinsic reward.
        """
        pred: Tensor = self.rnd_predictor(obses)
        target: Tensor = self.rnd_target(obses)
        intrinsic_reward: Tensor = (pred - target).pow(2)
        self.optimizer.zero_grad()
        intrinsic_reward.mean().backward()
        self.optimizer.step()
        return intrinsic_reward