from .flow_model import FlowModel
from .flow_model import FlowTransformLayer
from .flow_utils import deriv_tanh
from .flow_utils import FlowLayerStacker
import numpy as np
import numpy as ndarray
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import warnings

class PlanarTransformLayer(FlowTransformLayer):
    """transform layer of PlanarFlow.

    The following is the backward transform function f_k_(z_k) of PlanarFlow.
        f_k_(z_k) = z_k + tanh(w.T @ z_k + b) * u
    """
    def __init__(self, input_shape: ndarray):
        super(PlanarTransformLayer, self).__init__(input_shape)
        self.input_dim: int = np.prod(self.input_shape)
        self.register_parameter(
            "u", nn.Parameter(torch.randn(1, self.input_dim) * 0.01)
        )
        self.register_parameter(
            "w", nn.Parameter(torch.randn(1, self.input_dim) * 0.01)
        )
        self.register_parameter(
            "b", nn.Parameter(torch.randn(1) * 0.1)
        )
        self._make_invertible()

    @torch.no_grad()
    def _make_invertible(self) -> None:
        """make f_k invertible.

        f_k_ is invertible if -1 <= w.T @ u hold true. In this implementation,
        f_k_ is kept to be invertible by:
            u <- u + {m(w.T @ u) - w.T @ u} w / ||w||2,
            m(x) = -1 + log(1+e^x)
        """
        u: Tensor = self.u
        w: Tensor = self.w
        w_u: Tensor = torch.mm(u, w.t())
        if -1 < float(w_u):
            return
        u = u + (
            - 1 + F.softplus(w_u) - w_u
        ) * w / torch.mm(w, w.T)
        self.u.data = u
        w_u: Tensor = torch.mm(u, w.t())
        if float(w_u) <= -1:
            raise ValueError(
                f"failed to make f_k invertible. w.T@u={float(w_u)}."
            )

    @torch.no_grad()
    def is_invertible(self) -> bool:
        u: Tensor = self.u
        w: Tensor = self.w
        w_u: Tensor = torch.mm(u, w.t())
        if -1 < float(w_u):
            return True
        else:
            return False

    def _calc_g_res(
        self,
        a: Tensor,
        w_z_: Tensor,
        w_u: Tensor
    ) -> Tensor:
        g_res: Tensor = w_z_ - a - w_u * torch.tanh(a + self.b)
        return g_res

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """forward method.

        calculate z_k from z_{k-1} by solving numerically following equation.
            z_{k-1} = z_k + tanh(w.T @ z_k + b) * u

        follow the procedure below to find z_k.
        1. bimetrically solve the following equation for a.
            w.T @ z_{k-1} = a + w.T @ u * tanh(a + b)
        2. z_k = z_k_ - tanh(a+b) * u

        det Jacobian |df_k/dz_{k-1}| in PlanarFlow can be calculated by:
            |df_k/dz_{k-1}| = - (1 + tanh'(w.T @ z_k + b) * u.T @ w)

        Args:
            z_k_ (Tensor): k-1 th latent variables z_{k-1}. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k-1 ```[batch_size, 1]```.

        Returns:
            z_k [Tensor]: k th latent variables z_k. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k ```[batch_size, 1]```.
        """
        self._make_invertible()
        z_k_ = z_k_.view(-1, self.input_dim)
        w_z_: Tensor = torch.mm(z_k_, self.w.t())
        w_u: Tensor = torch.mm(self.u, self.w.t())
        with torch.no_grad():
            lo: Tensor = torch.full_like(w_z_, -1e+05)
            hi: Tensor = torch.full_like(w_z_, 1e+05)
            for _ in range(1000):
                a: Tensor = 0.5 * (lo + hi)
                g_res: Tensor = w_z_ - a + w_u * torch.tanh(a + self.b)
                lo: Tensor = torch.where(0 < g_res, a, lo)
                hi: Tensor = torch.where(g_res < 0, a, hi)
                if torch.max(
                    torch.abs(g_res)
                ) < 1e-08:
                    break
        if not torch.max(
            torch.abs(g_res)
        ) < 1e-08:
            warnings.warn(
                "numerical solver did not converge."
            )
        affine: Tensor = a + self.b
        z_k: Tensor = z_k_ - self.u * torch.tanh(affine)
        z_k = z_k.view(-1, *self.input_shape)
        log_det_jacobian = log_det_jacobian + torch.sum(
            - torch.log(torch.abs(1.0 + w_u * deriv_tanh(affine)) + 1e-10),
            dim=1
        )
        return z_k, log_det_jacobian

    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        f_k_(z_k) = z_k + tanh(w.T @ z_k + b) * u
        det Jacobian |df_k_/dz_k| in PlanarFlow can be calculated by:
            |df_k_/dz_k| = 1 + tanh'(w.T @ z_k + b) * u.T @ w.

        Args:
            z_k (Tensor): k th latent variables z_k. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian - sum(log|df_l_/dz_l|)), k+1<=l. ```[batch_size, 1]```.

        Returns:
            z_k_ [Tensor]: k-1 th latent variables z_{k-1}. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian - sum(log|df_l_/dz_l|)), k<=l. ```[batch_size, 1]```.
        """
        self._make_invertible()
        z_k = z_k.view(-1, self.input_dim)
        w_u: Tensor = torch.mm(self.u, self.w.t())
        affine: Tensor = torch.mm(z_k, self.w.t()) + self.b
        z_k_ = z_k + self.u * torch.tanh(affine)
        z_k_ = z_k_.view(-1, *self.input_shape)
        log_det_jacobian: Tensor = log_det_jacobian - torch.sum(
            torch.log(torch.abs(1.0 + w_u * deriv_tanh(affine)) + 1e-10),
            dim=1
        )
        return z_k_, log_det_jacobian

class PlanarFlow(FlowModel):
    def _create_network(self, config_dic: dict[str, int]) -> Module:
        if "num_layers" not in config_dic.keys():
            raise ValueError(
                "PlanarFlow requires 'num_layers' in config_dic."
            )
        num_layers: int = config_dic["num_layers"]
        layers: list[Module] = []
        for _ in range(num_layers):
            layers.append(PlanarTransformLayer(self.input_shape))
        net: Module = FlowLayerStacker(layers)
        return net