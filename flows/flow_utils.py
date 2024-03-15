from abc import abstractmethod
from .flow_model import FlowTransformLayer
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn.utils.parametrizations import weight_norm
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
        for layer in self.layers:
            latent_variables, log_det_jacobian = layer.backward(
                latent_variables, log_det_jacobian
            )
        return latent_variables, log_det_jacobian

class FlowBatchNorm(FlowTransformLayer):
    """Batch Normalization module for Flow layer."""
    def __init__(
        self,
        input_shape: ndarray,
        momentum: float = 0.1,
        is_affine_learnable: bool = True
    ) -> None:
        """initialization.

        apply Batch Normalization over a input tensor x by following calculation.

            z_ = gamma * (z - mean(z)) / std(z) + beta.
            mean and std are calculated per-dimension over mini-batches.
            gamma and beta are learnable parameter vectors.

        Also by default, during training this layer keeps running estimates
        of its computed mean and variance, which are then used for normalization during evaluation.
        The running estimates are kept with a default momentum of 0.1.

        Args:
            input_shape (ndarray): shape of input tensor.
            momentum (float, optional): the value used for the running_mean and running_var computation.
                Defaults to 0.1.
            is_affine_learnable (bool, optional): if set to True, this module has
                learnable affine parameters. Defaults to True.
        """
        super(FlowBatchNorm, self).__init__(input_shape)
        self.momentum: float = momentum
        self.input_shape_dim: list[int] = [1] + [1 for _ in input_shape]
        self.input_shape_dim[1] = input_shape[0] # [1, 1, 1, 1] w/ batch
        log_gamma: Tensor = torch.zeros(self.input_shape_dim)
        beta: Tensor = torch.zeros(self.input_shape_dim)
        if is_affine_learnable:
            self.register_parameter(
                "log_gamma", nn.Parameter(log_gamma)
            )
            self.register_parameter(
                "beta", nn.Parameter(beta)
            )
        else:
            self.register_buffer(
                "log_gamma", log_gamma
            )
            self.register_buffer(
                "beta", beta
            )
        self.register_buffer("running_mean", torch.zeros(self.input_shape_dim))
        self.register_buffer("running_var", torch.ones(self.input_shape_dim))
        self.register_buffer("batch_mean", torch.zeros(self.input_shape_dim))
        self.register_buffer("batch_var", torch.ones(self.input_shape_dim))

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var
        z_k_ = (z_k_ - self.beta) / torch.exp(self.log_gamma)
        z_k_ = z_k_ * torch.sqrt(var) + mean
        num_pixels: int = np.prod(
            z_k_.size()
        ) // (
            z_k_.size(0) * z_k_.size(1)
        )
        log_det_jacobian = log_det_jacobian + torch.sum(
            self.log_gamma - 0.5 * torch.log(var)
        ) * num_pixels
        return z_k_, log_det_jacobian

    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        Args:
            z_k (Tensor): k th latent variables z_k. ```[batch_size, *input_shape]```.
            log_det_jacobian (Optional[Tensor]): initial value of log det jacobian.

        Returns:
            z_k (Tensor): k th latent variables z_k. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of jacobian. ```batch_size, 1]```.
        """
        if self.training:
            z_k_reshape: Tensor = z_k.view(
                z_k.size(0), self.input_shape[0], -1
            )
            z_k_reshape_mean: Tensor = torch.mean(
                z_k_reshape, dim=[0,2], keepdim=True
            )
            z_k_reshape_var: Tensor = torch.mean(
                (z_k_reshape - z_k_reshape_mean).pow(2),
                dim=[0,2], keepdim=True
            ) + 1e-10
            self.batch_mean.data.copy_(
                z_k_reshape_mean.view(self.input_shape_dim)
            )
            self.batch_var.data.copy_(
                z_k_reshape_var.view(self.input_shape_dim)
            )
            self.running_mean.mul_(1.0 - self.momentum)
            self.running_var.mul_(1.0 - self.momentum)
            self.running_mean.add_(self.batch_mean.detach() * self.momentum)
            self.running_var.add_(self.batch_var.detach() * self.momentum)
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var
        z_k = (
            z_k - mean
        ) / torch.sqrt(var)
        z_k = z_k * torch.exp(self.log_gamma) + self.beta
        num_pixels: int = np.prod(
            z_k.size()
        ) // (
            z_k.size(0) * z_k.size(1)
        )
        log_det_jacobian = log_det_jacobian + torch.sum(
            self.log_gamma - 0.5 * torch.log(var)
        ) * num_pixels
        return z_k, log_det_jacobian

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
            weight_norm(nn.Linear(input_dim, output_dim)),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            weight_norm(nn.Linear(output_dim, output_dim))
        )
        if input_dim != output_dim:
            self.bridge: Module = weight_norm(
                nn.Linear(input_dim, output_dim)
            )
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
            weight_norm(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3+int(reduce_size), stride=1, padding=1
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            weight_norm(
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1
                )
            )
        )
        if in_channels != out_channels:
            self.bridge: Module = weight_norm(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1
                )
            )
        else:
            self.bridge: Module = nn.Sequential()

    def forward(self, input_tensor: Tensor):
        output_tensor: Tensor = self.net(input_tensor)
        output_tensor += self.bridge(input_tensor)
        return output_tensor

