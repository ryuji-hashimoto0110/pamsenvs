import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

def initialize_module_orthogonal(module: Module) -> None:
    for name, param in module.named_parameters():
        device: torch.device = param.device
        with torch.no_grad():
            if "weight" in name:
                tmp = torch.empty_like(param, device="cpu")
                nn.init.orthogonal_(tmp)
            elif "bias" in name:
                nn.init.zeros_(param)

def calc_log_prob(
    log_stds: Tensor,
    noises: Tensor,
    actions: Tensor,
) -> Tensor:
    gaussian_log_probs: Tensor = (
        -0.5 * noises.pow(2) - log_stds
    ).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    log_probs: Tensor = gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-06
    ).sum(dim=-1, keepdim=True)
    return log_probs

def reparametrize(
    means: Tensor, log_stds: Tensor,
) -> tuple[Tensor]:
    stds: Tensor = log_stds.exp()
    noises: Tensor = torch.randn_like(means)
    us: Tensor = means + noises * stds
    actions: Tensor = torch.tanh(us)
    log_probs: Tensor = calc_log_prob(log_stds, noises, actions)
    return actions, log_probs