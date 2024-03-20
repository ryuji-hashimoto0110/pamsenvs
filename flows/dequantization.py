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
        if "randon_std" in config_dic.keys():
            self.randn_std: float = config_dic["randn_std"]
        self.activate_func: Optional[str] = None
        if "activate_func" in config_dic.keys():
            if config_dic["activate_func"] == "tanh":
                self.activate_func: str = "tanh"
            else:
                raise NotImplementedError

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b: int = z_k_.shape[0]
        z_k: Tensor = z_k_
        if self.activate_func == "tanh":
            z_k = torch.tanh(z_k_)
            log_det_jacobian = log_det_jacobian + torch.sum(
                torch.log(
                    deriv_tanh(z_k_.view(b,-1))
                ),
                dim=1
            )
        return z_k, log_det_jacobian

    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b: int = z_k.shape[0]
        z_k_: Tensor = z_k
        if self.activate_func == "tanh":
            z_k = torch.clamp(z_k, -0.99999, 0.99999)
            z_k_: Tensor = 0.5 * torch.log(
                (1 + z_k) / (1 - z_k)
            )
            log_det_jacobian = log_det_jacobian + torch.sum(
                torch.log(
                    1 - z_k.view(b,-1).pow_(2)
                ),
                dim=1
            )
        if self.training:
            z_k_ = z_k_ + self.randn_std * torch.randn_like(z_k_)
        return z_k_, log_det_jacobian