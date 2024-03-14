from abc import abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Module
from typing import Literal
from typing import Optional

class FlowTransformLayer(Module):
    """flow transform layer f_k.

    f_k must be invertible and should be easy to calculate det Jacobian.
    """
    def __init__(self, input_shape: ndarray) -> None:
        super().__init__()
        self.input_shape: ndarray = input_shape

    @abstractmethod
    def forward(
        self,
        z_k_: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """forward method.

        Args:
            z_k_ (Tensor): k-1 th latent variables z_{k-1}. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k-1 ```[batch_size, 1]```.

        Returns:
            z_k [Tensor]: k th latent variables z_k. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l/dz_{l-1}|), l<=k ```[batch_size, 1]```.
        """
        pass

    @abstractmethod
    def backward(
        self,
        z_k: Tensor,
        log_det_jacobian: Tensor
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        Args:
            z_k (Tensor): k th latent variables z_k. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_l_/dz_l|)), k+1<=l. ```[batch_size, 1]```.

        Returns:
            z_k_ [Tensor]: k-1 th latent variables z_{k-1}. ```[batch_size, input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian - sum(log|df_l_/dz_l|)), k<=l. ```[batch_size, 1]```.
        """
        pass

class FlowModel(Module):
    """
    FlowModel class is the base of any normalizing flow algorithm.

    FlowModel is one of the latent variable model whose flow can be written as:
        z_0 <-> z_1 <-> ... z_k_ <-> z_k <-> ... <-> z_K == x.
        x = f_K * ... * f_1 (z_0).
        z_0 = f_1_ * ... * f_K_ (x). (f_k_ denotes the inverse function of f_k.)
    """
    def __init__(
        self,
        config_dic: dict[str, int]
    ) -> None:
        """initialization.

        Args:
            config_dic (dict[str, int]): config dictionary.
        """
        super().__init__()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if "input_shape" not in config_dic.keys():
            raise ValueError(
                "FlowModel requires 'input_shape' in config_dic."
            )
        self.input_shape: ndarray = np.array(
            config_dic["input_shape"], dtype=np.uint8
        )
        self.input_dim: int = np.prod(self.input_shape)
        mu = torch.zeros(
            self.input_dim, dtype=torch.float32, device=self.device
        )
        cov = torch.eye(
            self.input_dim, dtype=torch.float32, device=self.device
        )
        self.register_buffer(
            "mu", mu
        )
        self.register_buffer(
            "cov", cov
        )
        self.register_buffer(
            "inv_cov", torch.linalg.inv(cov)
        )
        self.normal_dist = MultivariateNormal(self.mu, self.cov)
        self.net: Module = self._create_network(config_dic)

    @abstractmethod
    def _create_network(self, config_dic: dict[str, int]) -> Module:
        """create flow network.

        create network that corresponds to the flow algorithm.
        the network has .forward and .backward methods.
        It is reccommended to use FlowTransformLayer when implementation.

        Args:
            config_dic (dict[str, int]): config dictionary.

        Returns:
            net (Module): flow network.
        """
        pass

    @abstractmethod
    def forward(
        self,
        latent_variables: Tensor
    ) -> tuple[Tensor, Tensor]:
        """forward method.

        transform initial latent variable z_0 to observed variable x=z_K by
        multiple, invertible and nonlinear functions: f_K * ... * f_1 (z_0).

        Args:
            latent_variables (Tensor): latent variables z_0. ```[batch_size, *input_shape]```.

        Returns:
            observed_variable (Tensor): observed variebles x. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_k/dz_{k-1}|). ```[batch_size, 1]```.
        """
        pass

    @abstractmethod
    def backward(
        self,
        observed_variables: Tensor
    ) -> tuple[Tensor, Tensor]:
        """backward method.

        transform abserved variable x=z_K to latent variable z_0 by
        multiple inverted functions: f_1_ * ... * f_K_ (x).

        Args:
            observed_variables (Tensor): observed varieble x. ```[batch_size, *input_shape]```.

        Returns:
            latent_variables (Tensor): latent variable z_0. ```[batch_size, *input_shape]```.
            log_det_jacobian (Tensor): log determinant of
                jacobian -sum(log|df_k_ /dz_k|). ```[batch_size, 1]```.
        """
        pass

    def calc_log_likelihood(
        self,
        observed_variables: Optional[Tensor] = None,
        latent_variables: Optional[Tensor] = None,
        reduction: Literal["none", "joint"] = "none"
    ) -> Tensor:
        """calculate log likelihood.

        either of observed_variable or latent_variable must be Tensor.

        log likelihood of normalizing flow can be calculated either by
            log p(x) = log Normal(z_0) - sum_k (
                log |df_k / dz_{k-1}|
            )
            log p(x) = log Normal(z_0) + sum_k (
                log |df_k_ / dz_k|
            )


        Args:
            observed_variables (Optional[Tensor]):
                observed varieble x. ```[batch_size, *input_shape]```.
                default to None.
            latent_variables (Optional[Tensor]):
                latent variable z_0. ```[batch_size, *input_shape]```.
                default to None.

        Returns:
            log_likelihoods (Tensor): _description_
        """
        if observed_variables is not None:
            if latent_variables is not None:
                raise ValueError(
                    "both observed_variable and latent_variable found not None."
                )
            latent_variables, log_det_jacobians = self.backward(observed_variables)
            if reduction == "none":
                log_likelihoods: Tensor = self.normal_dist.log_prob(
                    latent_variables.view(-1, self.input_dim)
                ) - log_det_jacobians
            elif reduction == "joint":
                log_likelihoods: Tensor = self._calc_joint_log_prob(
                    latent_variables.view(-1, self.input_dim)
                ) - torch.sum(log_det_jacobians)
            else:
                raise NotImplementedError
        elif latent_variables is not None:
            if reduction == "none":
                log_likelihoods: Tensor = self.normal_dist.log_prob(
                    latent_variables.view(-1, self.input_dim)
                )
            elif reduction == "joint":
                log_likelihoods: Tensor = self._calc_joint_log_prob(
                    latent_variables.view(-1, self.input_dim)
                )
            else:
                raise NotImplementedError
        else:
            raise ValueError(
                "either of observed_variable or latent_variable must be Tensor."
            )
        return log_likelihoods

    def _calc_joint_log_prob(
        self,
        latent_variables: Tensor
    ) -> Tensor:
        assert latent_variables.size(1) == self.input_dim
        num_samples, input_dim = latent_variables.shape
        log_likelihood: Tensor = - num_samples * input_dim * \
            torch.log(2*torch.tensor([torch.pi])) / 2
        log_likelihood -= num_samples * torch.log(torch.linalg.det(self.cov)) / 2
        log_likelihood -= torch.sum(
            torch.diagonal(
                (latent_variables - self.mu) @ self.inv_cov @ \
                    (latent_variables - self.mu).T
            )
        ) / 2
        return log_likelihood

    def sample_latent_variables(self, num_samples: int) -> Tensor:
        """sample arbitrary number of latent variables

        Args:
            num_samples (int): number of samples

        Returns:
            latent_variables (Tensor): _description_
        """
        latent_variables: Tensor = self.normal_dist.sample(
            [num_samples]
        ).view(num_samples, *self.input_shape)
        return latent_variables

    @torch.no_grad()
    def sample_observed_variables(
        self,
        num_samples: int
    ) -> Tensor:
        """sample arbitrary number of observed variables

        Args:
            num_samples (int): number of samples

        Returns:
            latent_variables (Tensor): _description_
            log_det_jacobian (Tensor): log determinant of
                jacobian sum(log|df_k/dz_{k-1}|). ```batch_size, 1]```.
        """
        latent_variables: Tensor = self.sample_latent_variables(num_samples)
        observed_variables, log_det_jacobian = self.forward(latent_variables)
        return observed_variables, log_det_jacobian
