from abc import abstractmethod
import copy
from .flow_model import FlowTransformLayer
from .flow_utils import ConvResBlock
from .flow_utils import LinearResBlock
from numpy import ndarray
import torch
from torch import nn
from torch.nn import Module
from torch import Tensor
from typing import Literal

def split1d(
    z: Tensor,
    is_odd: bool
) -> tuple[Tensor]:
    """split 1d latent variables.

    Args:
        z (Tensor): 1d latent variable. ```[batch_size, input_dim]```
        is_odd (bool): wether the number of the layer in the flow layers is odd.
    """
    if z.dim() != 2:
        raise ValueError(
            f"split1d got z whose shape is {z.shape}."
        )
    _, input_dim = z.shape
    z1 = z[:,:input_dim//2]
    z2 = z[:,input_dim//2:]
    if is_odd:
        z1, z2 = z2, z1
    return z1, z2

def merge1d(
    z1: Tensor,
    z2: Tensor,
    is_odd: bool
) -> Tensor:
    """backward transformation of split1d function.
    """
    if z1.dim() != 2 or z2.dim() != 2:
        raise ValueError(
            "merge1d got tensor whose shape is" +
            f"z1.shape={z1.shape}, z2.shape={z2.shape}"
        )
    assert z1.shape[0] == z2.shape[0]
    if is_odd:
        z1, z2 = z2, z1
    z: Tensor = torch.cat([z1, z2], dim=1).contiguous()
    return z

def mask_checker(
    z1: Tensor,
    z2: Tensor
) -> tuple[Tensor]:
    masked_z1 = z1.clone()
    masked_z2 = z2.clone()
    masked_z1[:,:,0::2,1::2] = 0
    masked_z1[:,:,1::2,0::2] = 0
    masked_z2[:,:,0::2,0::2] = 0
    masked_z2[:,:,1::2,1::2] = 0
    return masked_z1, masked_z2

def split_checker(
    z: Tensor,
    is_odd: bool
) -> tuple[Tensor]:
    """split image shaped latent variables with spatial checkerboard pattern.

    Args:
        z (Tensor): latent variable. ```[batch_size, c, h, w]```
        is_odd (bool): wether the number of the layer in the flow layers is odd.
    """
    if z.dim() != 4:
        raise ValueError(
            f"z must be image-shaped. z.shape={z.shape}"
        )
    z1: Tensor = z.clone().contiguous()
    z2: Tensor = z.clone().contiguous()
    masked_z1, masked_z2 = mask_checker(z1, z2)
    if is_odd:
        masked_z1, masked_z2 = masked_z2, masked_z1
    return masked_z1, masked_z2

def merge_checker(
    z1: Tensor, z2: Tensor,
    is_odd: bool
) -> Tensor:
    """backward transformation of split_checker function.
    """
    if z1.dim() != 4 or z2.dim() != 4:
        raise ValueError(
            "merge_checker got tensor whose shape is" +
            f"z1.shape={z1.shape}, z2.shape={z2.shape}"
        )
    if is_odd:
        z1, z2 = z2, z1
    masked_z1, masked_z2 = mask_checker(z1, z2)
    z: Tensor = masked_z1 + masked_z2
    return z

def split_channel(
    z: Tensor,
    is_odd: bool
) -> tuple[Tensor]:
    """split image shaped latent variables with channel-wise masking.

    Args:
        z (Tensor): latent variable. ```[batch_size, c, h, w]```
        is_odd (bool): wether the number of the layer in the flow layers is odd.
    """
    if z.dim() != 4:
        raise ValueError(
            f"z must be image-shaped. z.shape={z.shape}"
        )
    c: int = z.shape[1]
    assert 2 <= c
    z1: Tensor = z[:,:c//2,:,:].contiguous()
    z2: Tensor = z[:,c//2:,:,:].contiguous()
    if is_odd:
        z1, z2 = z2, z1
    return z1, z2

def merge_channel(
    z1: Tensor, z2: Tensor,
    is_odd: bool
) -> Tensor:
    """backward transformation of split_channel function.
    """
    if z1.dim() != 4 or z2.dim() != 4:
        raise ValueError(
            "merge_channel got tensor whose shape is" +
            f"z1.shape={z1.shape}, z2.shape={z2.shape}"
        )
    b1, _, h1, w1 = z1.shape
    b2, _, h2, w2 = z2.shape
    assert b1 == b2 and h1 == h2 and w1 == w2
    if is_odd:
        z1, z2 = z2, z1
    z: Tensor = torch.cat([z1, z2], dim=1).contiguous()
    return z

class Squeeze1dLayer(FlowTransformLayer):
    def __init__(
        self,
        input_shape: ndarray,
    ) -> None:
        super(Squeeze1dLayer, self).__init__(input_shape)

class Squeeze2dLayer(FlowTransformLayer):
    def __init__(
        self,
        input_shape: ndarray,
    ) -> None:
        super(Squeeze2dLayer, self).__init__(input_shape)

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b, c, h, w = self.input_shape
        if c % 4 != 0:
            raise ValueError(
                "the number of channels c must be c=0 (mod 4). " +
                f"z_k.shape={z_k.shape}"
            )
        z_k = z_k_.view(b, c//4, 2, 2, h, w)
        z_k = z_k.permute(0,1,4,2,5,3).contiguous()
        z_k = z_k.view(b, c//4, 2*h, 2*w)
        return z_k, log_det_jacobian

    def backward(
        self, z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b, c, h, w = z_k.shape
        if (h % 2 != 0) or (w % 2 != 0):
            raise ValueError(
                "the shape of input image must have even height and width. " +
                f"z_k.shape={z_k.shape}"
            )
        z_k_ = z_k.view(b, c, h//2, 2, w//2, 2)
        z_k_ = z_k_.permute(0,1,3,5,2,4).contiguous()
        z_k_ = z_k_.view(b, 4*c, h//2, w//2)
        assert z_k_.shape == self.input_shape
        return z_k_, log_det_jacobian

class ReshapeLayer(FlowTransformLayer):
    def __init__(
        self,
        input_shape: ndarray,
        output_shape: ndarray,
    ) -> None:
        super(ReshapeLayer, self).__init__(input_shape)
        self.output_shape = output_shape

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b = z_k_.shape[0]
        z_k = z_k_.view(b, *self.output_shape)
        return z_k, log_det_jacobian

    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        b = z_k.shape[0]
        z_k_ = z_k.view(b, *self.input_shape)
        return z_k_, log_det_jacobian

class BijectiveCouplingLayer(FlowTransformLayer):
    """abstract class for bijective coupling layers
    """
    def __init__(
        self,
        input_shape: ndarray,
        split_pattern: Literal["checkerboard", "channelwise"] = "checkerboard",
        is_odd: bool = False
    ) -> None:
        """initialization.

        Bijective Coupling (BC) is a neural net architecture that is bijective
        and is easy to calculate determinants of Jacobian matrix.

        Args:
            input_shape (ndarray): the shape of input tensor
            split_pattern (str): how to split tensor.
            is_odd (bool, optional): wether the number of the layer in the flow layers is odd.
        """
        super(BijectiveCouplingLayer, self).__init__(input_shape)
        if len(self.input_shape) == 1:
            self.split = lambda z, is_odd=is_odd: split1d(z, is_odd)
            self.merge = lambda z1, z2, is_odd=is_odd: merge1d(z1, z2, is_odd)
        elif (
            len(self.input_shape) == 3 and
            split_pattern == "checkerboard"
        ):
            self.split = lambda z, is_odd=is_odd: split_checker(z, is_odd)
            self.merge = lambda z1, z2, is_odd=is_odd: merge_checker(z1, z2, is_odd)
        elif (
            len(self.input_shape) == 3 and
            split_pattern == "channelwise"
        ):
            self.split = lambda z, is_odd=is_odd: split_channel(z, is_odd)
            self.merge = lambda z1, z2, is_odd=is_odd: merge_channel(z1, z2, is_odd)
        else:
            raise NotImplementedError

    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """forward method.

        Args:
            z_k_ (Tensor): k-1 th latent variables z_{k-1}. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k-1 ```[batch_size, 1]```.

        Returns:
            z_k [Tensor]: k th latent variables z_k. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k ```[batch_size, 1]```.
        """
        z_k1_, z_k2_ = self.split(z_k_)
        z_k1, z_k2, log_det_jacobian = self._transform(
            z_k1_, z_k2_, log_det_jacobian
        )
        z_k = self.merge(z_k1, z_k2)
        return z_k, log_det_jacobian

    def backward(
        self, z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        Args:
            z_k (Tensor): k th latent variables z_k. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian - sum(log|df_l_/dz_l|)), k+1<=l. ```[batch_size, 1]```.

        Returns:
            z_k_ [Tensor]: k-1 th latent variables z_{k-1}. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian - sum(log|df_l_/dz_l|)), k<=l. ```[batch_size, 1]```.
        """
        z_k1, z_k2 = self.split(z_k)
        z_k1_, z_k2_, log_det_jacobian = self._inverse_transform(
            z_k1, z_k2, log_det_jacobian
        )
        z_k_ = self.merge(z_k1_, z_k2_)
        return z_k_, log_det_jacobian

    @abstractmethod
    def _transform(
        self,
        z_k1_: Tensor,
        z_k2_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor]:
        """transform method.

        BC layer f_k transforms input blocks z_k1_ and z_k2_ into z_k1 and z_k2.
            z_k = f_k(z_k_)
            z_k1_, z_k2_ = split(z_k_)
            z_k1 = z_k1_
            z_k2 = g(z_k2_ | phi(z_k1_))
            z_k = merge(z_k1, z_k2)

        Args:
            z_k1_ (Tensor):
            z_k2_ (Tensor):
            log_det_jacobian (Tensor):

        Returns:
            z_k1 (Tensor)
            z_k2 (Tensor)
            log_det_jacobian (Tensor)
        """
        pass

    @abstractmethod
    def _inverse_transform(
        self,
        z_k1: Tensor,
        z_k2: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor]:
        """inverse transform method.

        BC layer f_k^{-1} transforms input blocks z_k1 and z_k2 into z_k1_ and z_k2_.
            z_k_ = f_k^{-1}(z_k)
            z_k1, z_k2 = split(z_k)
            z_k1_ = z_k1
            z_k2_ = g^{-1}(z_k2 | phi(z_k1))
            z_k_ = merge(z_k1_, z_k2_)

        Args:
            z_k1 (Tensor): _description_
            z_k2 (Tensor): _description_
            log_det_jacobian (Tensor): _description_

        Returns:
            z_k1_ (Tensor): _description_
            z_k2_ (Tensor): _description_
            log_det_jacobian (Tensor): _description_
        """
        pass

class AffineCouplingLayer(BijectiveCouplingLayer):
    def __init__(
        self,
        input_shape: ndarray,
        split_pattern: Literal["checkerboard", "channelwise"] = "checkerboard",
        is_odd: bool = False
    ) -> None:
        super(AffineCouplingLayer, self).__init__(
            input_shape, split_pattern, is_odd
        )
        self.input_shape = input_shape
        self.split_pattern = split_pattern
        self.register_parameter(
            "s_log_scale", nn.Parameter(torch.randn(1) * 0.01)
        )
        self.register_parameter(
            "s_bias", nn.Parameter(torch.randn(1) * 0.01)
        )
        if len(input_shape) == 1:
            input_dim: int = input_shape[0] // 2 if not is_odd else (input_shape[0] + 1) // 2
            output_dim: int = input_shape[0] - input_dim
            self.net: Module = LinearResBlock(
                input_dim=input_dim, output_dim=output_dim*2
            )
            self.out_channels: int = output_dim
        elif len(input_shape) == 3:
            if split_pattern == "checkerboard":
                in_channels: int = input_shape[0]
                self.out_channels: int = in_channels
            elif split_pattern == "channelwise":
                in_channels: int = input_shape[0] // 2 if not is_odd else (input_shape[0] + 1) // 2
                self.out_channels: int = input_shape[0] - in_channels
            reduce_size: bool = False
            self.net: Module = ConvResBlock(
                in_channels=in_channels, out_channels=self.out_channels*2,
                reduce_size=reduce_size
            )
        else:
            raise NotImplementedError

    def _transform(
        self,
        z_k1_: Tensor,
        z_k2_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor]:
        """_transform method.

        Affine Coupling transforms input blocks z_k1_ and z_k2_ into z_k1 and z_k2 by the
        following calculation with s, t, which stand for scale and translation.
            z_k1 = z_k1_
            z_k2 = z_k2_ * exp(s(z_k1_)) + t(z_k1_)

        det Jacobian |df_k/dz_{k-1}| in AffineCoupling can be calculated by:
            |df_k/dz_{k-1}| = sum exp(s(z_k1_))

        Args:
            z_k1_ (Tensor): _description_
            z_k2_ (Tensor): _description_
            log_det_jacobian (Tensor): _description_

        Returns:
            z_k1 (Tensor)
            z_k2 (Tensor)
            log_det_jacobian
        """
        t_z_k1_: Tensor = self.net(z_k1_)
        s_z_k1_: Tensor = torch.tanh(
            t_z_k1_[:,:t_z_k1_.shape[1]//2]
        ) * self.s_log_scale + self.s_bias
        t_z_k1_ = t_z_k1_[:,t_z_k1_.shape[1]//2:]
        z_k1: Tensor = z_k1_
        z_k2: Tensor = z_k2_ * torch.exp(s_z_k1_) + t_z_k1_
        log_det_jacobian = log_det_jacobian + torch.sum(
            s_z_k1_.view(s_z_k1_.shape[0], -1),
            dim=1
        )
        return z_k1, z_k2, log_det_jacobian

    def _inverse_transform(
        self,
        z_k1: Tensor,
        z_k2: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor]:
        """_inverse_transform method.

        inverse transformation of Affine Coupling transforms blocks z_k1 and z_k2
        into z_k1_ and z_k2_.
            z_k1_ = z_k1
            z_k2_ = (z_k2 - t(z_k1)) * exp(-s(z_k1))

        Args:
            z_k1 (Tensor): _description_
            z_k2 (Tensor): _description_
            log_det_jacobian (Tensor): _description_

        Returns:
            z_k1_ (Tensor)
            z_k2_ (Tensor)
            log_det_jacobian
        """
        t_z_k1: Tensor = self.net(z_k1)
        s_z_k1: Tensor = torch.tanh(
            t_z_k1[:,:t_z_k1.shape[1]//2]
        ) * self.s_log_scale + self.s_bias
        t_z_k1 = t_z_k1[:,t_z_k1.shape[1]//2:]
        z_k1_: Tensor = z_k1
        z_k2_: Tensor = (z_k2 - t_z_k1) * torch.exp(-s_z_k1)
        log_det_jacobian = log_det_jacobian + torch.sum(
            s_z_k1.view(s_z_k1.shape[0], -1),
            dim=1
        )
        return z_k1_, z_k2_, log_det_jacobian