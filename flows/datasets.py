from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset

class CircleDataset2d(Dataset):
    def __init__(
        self,
        radius: float,
        center: list[float] | ndarray,
        randn_std: float,
        num_sample: int
    ) -> None:
        angles_tensor: Tensor = 2 * torch.pi * torch.rand(num_sample)
        x_tensor = radius * torch.cos(angles_tensor) + center[0] + \
            torch.randn_like(angles_tensor) * randn_std
        y_tensor = radius * torch.sin(angles_tensor) + center[1] + \
            torch.randn_like(angles_tensor) * randn_std
        self.points_tensor = torch.stack([x_tensor, y_tensor], dim=1)

    def __len__(self) -> int:
        return len(self.points_tensor)

    def __getitem__(self, index) -> Tensor:
        return self.points_tensor[index]