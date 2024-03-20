from abc import abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from typing import Optional

def deriv_tanh(x: float) -> float:
    """derivative of tanh"""
    return 1.0 - torch.tanh(x)**2

class FlowLayerStacker(Module):
    """stack flow layers."""
    def __init__(self, layers: list[Module]):
        """initialization

        Args:
            layers (list[Module]): list of flow layers. [f_1, ..., f_K].
                each layer must have .forward and backward method.
        """
        super(FlowLayerStacker, self).__init__()
        self.layers: ModuleList = ModuleList(layers)

    def forward(
        self,
        latent_variables: Tensor,
        log_det_jacobian: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """forward method.

        transform initial latent variable z_0 to observed variable x=z_K by
        multiple, invertible and nonlinear functions: f_K * ... * f_1 (z_0).

        Args:
            latent_variable (Tensor): latent variables z_0. ```[batch_size, *input_shape]```.
            log_det_jacobian (Optional[Tensor]): initial value of log det jacobian.

        Returns:
            observed_variable (Tensor): observed variebles x. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_k/dz_{k-1}|). ```batch_size, 1]```.
        """
        if log_det_jacobian is None:
            log_det_jacobian: Tensor = torch.zeros(
                latent_variables.shape[0]
            ).type_as(latent_variables).to(latent_variables.device)
        for layer in self.layers:
            latent_variables, log_det_jacobian = layer(
                latent_variables, log_det_jacobian
            )
        observed_variables: Tensor = latent_variables
        return observed_variables, log_det_jacobian

    def backward(
        self,
        observed_variables: Tensor,
        log_det_jacobian: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        transform abserved variable x=z_K to latent variable z_0 by
        multiple inverted functions: f_1_ * ... * f_K_ (x).

        Args:
            observed_variables (Tensor): observed varieble x. ```[batch_size, *input_shape]```.

        Returns:
            latent_variables (Tensor): latent variable z_0. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_k/dz_{k-1}|). ```batch_size, 1]```.
        """
        if log_det_jacobian is None:
            log_det_jacobian: Tensor = torch.zeros(
                observed_variables.shape[0]
            ).type_as(observed_variables).to(observed_variables.device)
        latent_variables: Tensor = observed_variables
        for layer in reversed(self.layers):
            latent_variables, log_det_jacobian = layer.backward(
                latent_variables, log_det_jacobian
            )
        return latent_variables, log_det_jacobian

class LinearResBlock(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super(LinearResBlock, self).__init__()
        self.net: Module = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        if input_dim != output_dim:
            self.bridge: Module = nn.Linear(input_dim, output_dim)
        else:
            self.bridge: Module = nn.Sequential()

    def forward(self, input_tensor: Tensor):
        output_tensor: Tensor = self.net(input_tensor)
        output_tensor += self.bridge(input_tensor)
        return output_tensor

class ConvResBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduce_size: bool
    ) -> None:
        super(ConvResBlock, self).__init__()
        self.net: Module = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3+int(reduce_size), stride=1, padding=1
            )
            ,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1
            )
        )
        if in_channels != out_channels:
            self.bridge: Module = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3+int(reduce_size),
                stride=1, padding=1
            )
        else:
            self.bridge: Module = nn.Sequential()

    def forward(self, input_tensor: Tensor):
        output_tensor: Tensor = self.net(input_tensor)
        output_tensor += self.bridge(input_tensor)
        return output_tensor

