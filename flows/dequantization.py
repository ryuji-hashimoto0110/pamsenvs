from .flow_model import FlowTransformLayer
from .flow_utils import deriv_tanh
from numpy import ndarray
import torch
from torch import Tensor

class DequantizationLayer(FlowTransformLayer):
    def __init__(
        self,
        input_shape: ndarray,
    ) -> None:
        super(DequantizationLayer, self).__init__(input_shape)

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b: int = z_k_.shape[0]
        z_k: Tensor = torch.tanh(z_k_)
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
        z_k = torch.clamp(z_k, -0.99999, 0.99999)
        z_k_: Tensor = 0.5 * torch.log(
            (1 + z_k) / (1 - z_k)
        )
        if self.training:
            z_k_ += torch.randn_like(z_k_)
        log_det_jacobian = log_det_jacobian + torch.sum(
            torch.log(
                1 - z_k.view(b,-1).pow_(2)
            ),
            dim=1
        )
        return z_k_, log_det_jacobian