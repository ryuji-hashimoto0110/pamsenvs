from .flow_model import FlowTransformLayer
from .flow_utils import deriv_tanh
from numpy import ndarray
import torch
from torch import Tensor
from typing import Optional

class DequantizationLayer(FlowTransformLayer):
    def __init__(
        self,
        input_shape: ndarray,
        config_dic: dict
    ) -> None:
        super(DequantizationLayer, self).__init__(input_shape)
        self.randn_std: float = 1.0
        if "randn_std" in config_dic.keys():
            self.randn_std: float = config_dic["randn_std"]
        self.activate_func: Optional[str] = None
        if "activate_func" in config_dic.keys():
            if config_dic["activate_func"] == "tanh":
                self.activate_func: str = "tanh"
            else:
                raise NotImplementedError
        self.random_noise: Optional[Tensor] = None

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b: int = z_k_.shape[0]
        if self.activate_func == "tanh":
            z_k = torch.tanh(z_k_)
            log_det_jacobian = log_det_jacobian + torch.sum(
                torch.log(
                    deriv_tanh(z_k_.view(b,-1))
                ),
                dim=1
            )
        else:
            z_k = z_k_
        if (
            self.training and
            self.random_noise is not None
        ):
            z_k = z_k - self.random_noise
        return z_k, log_det_jacobian

    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b: int = z_k.shape[0]
        if self.training:
            self.random_noise: Tensor = self.randn_std * torch.randn_like(z_k_)
            z_k = z_k + self.random_noise
        if self.activate_func == "tanh":
            z_k = torch.clamp(z_k, -0.999, 0.999)
            z_k_: Tensor = 0.5 * torch.log(
                (1 + z_k) / (1 - z_k)
            )
            log_det_jacobian = log_det_jacobian + torch.sum(
                torch.log(
                    deriv_tanh(z_k_.view(b,-1))
                ),
                dim=1
            )
        else:
            z_k_ = z_k
        return z_k_, log_det_jacobian