from .flow_model import FlowModel
import numpy as np
from pathlib import Path
import random
from rich import print
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional

class FlowTrainer:
    def __init__(
        self,
        model: FlowModel,
        recon_coef: float,
        optimizer_type: str,
        lr: float,
        weight_decay: float,
        train_dataset: Optional[Dataset],
        valid_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        num_epochs: int,
        batch_size: int,
        num_workers: int,
        store_recon_loss: bool = True,
        load_path: Optional[Path] = None,
        best_save_path: Optional[Path] = None,
        last_save_path: Optional[Path] = None,
        seed_range: list[int] = [42, 43]
    ) -> None:
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: FlowModel = model.to(self.device)
        self.recon_coef: float = recon_coef
        self.optimizer_type: str = optimizer_type
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.train_dataset = train_dataset
        if train_dataset is not None:
            self.train_n: int = len(train_dataset)
        self.valid_dataset = valid_dataset
        if valid_dataset is not None:
            self.valid_n: int = len(valid_dataset)
        self.test_dataset = test_dataset
        if test_dataset is not None:
            self.test_n: int = len(test_dataset)
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.load_path: Optional[Path] = load_path
        self.store_recon_loss: bool = store_recon_loss
        self.best_save_path: Optional[Path] = best_save_path
        self.last_save_path: Optional[Path] = last_save_path
        self.seed_range: list[int] = seed_range

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    def _setup(self, seed: int) -> None:
        if self.load_path is not None:
            load_path = \
                self.load_path.parent / f"{self.load_path.stem}_{seed}.pth"
            checkpoint = torch.load(
                load_path, map_location=torch.device(self.device)
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.start_epoch: int = checkpoint["current_epoch"]
            self.end_epoch: int = self.start_epoch + self.num_epochs
            self.train_losses: list[float] = checkpoint["train_losses"]
            self.train_recon_losses: list[float] = checkpoint["train_recon_losses"]
            self.valid_losses: list[float] = checkpoint["valid_losses"]
            self.valid_recon_losses: list[float] = checkpoint["valid_recon_losses"]
            self.best_loss: float = checkpoint["best_loss"]
        else:
            self.start_epoch: int = 0
            self.end_epoch: int = self.num_epochs
            self.train_losses: list[float] = []
            self.train_recon_losses: list[float] = []
            self.valid_losses: list[float] = []
            self.valid_recon_losses: list[float] = []
            self.best_loss: float = 1e+10
            self.model._init_weights()
        if self.optimizer_type == "Adam":
            self.optimizer: Optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay
            )

    def fit(self) -> None:
        print(
            f"start training. device: {self.device} " +
            f"number of train samples: {self.train_n} " +
            f"valid samples: {self.valid_n}"
        )
        assert self.train_dataset is not None
        assert self.valid_dataset is not None
        for seed in range(self.seed_range[0], self.seed_range[1]):
            print(f"seed: {seed}")
            self.set_seed(seed)
            self._setup(seed)
            self._create_dataloaders(self.batch_size, self.num_workers)
            for epoch in range(self.start_epoch, self.end_epoch):
                self.train_loss: float = 0.0
                self.train_recon_loss: float = 0.0
                for batch in self.train_dataloader:
                    self._train_step(batch)
                self.valid_loss: float = 0.0
                self.valid_recon_loss: float = 0.0
                for batch in self.valid_dataloader:
                    self.valid_loss, self.valid_recon_loss = self._validate_step(
                        batch, self.valid_loss, self.valid_recon_loss
                    )
                self.train_loss = self.train_loss / self.train_n
                self.train_recon_loss = self.train_recon_loss / self.train_n
                self.valid_loss = self.valid_loss / self.valid_n
                self.valid_recon_loss = self.valid_recon_loss / self.valid_n
                self._record_losses(epoch)
                self._save_checkpoint(epoch, seed)

    def _train_step(
        self,
        batch: Tensor | tuple[Tensor, Tensor]
    ) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        if isinstance(batch, Tensor):
            observed_variables: Tensor = batch.to(self.device)
        elif isinstance(batch, tuple) or isinstance(batch, list):
            observed_variables: Tensor = batch[0].to(self.device)
        else:
            raise ValueError(
                "batch must be either Tensor or tuple[Tensor]."
            )
        loss: Tensor = - self.model.calc_log_likelihood(
            observed_variables=observed_variables,
            reduction="joint",
            is_by_bit=True
        )
        latent_variables, _ = self.model.backward(observed_variables)
        observed_variables_, _ = self.model(latent_variables)
        recon_loss = torch.sum(
            (observed_variables - observed_variables_).pow_(2),
        )
        loss += self.recon_coef * recon_loss
        F.softplus(loss).backward()
        self.optimizer.step()
        self.train_loss += float(loss)
        self.train_recon_loss += float(recon_loss)

    @torch.no_grad()
    def _validate_step(
        self,
        batch: Tensor | tuple[Tensor, Tensor],
        valid_loss: float,
        recon_loss: Optional[float] = None
    ) -> float:
        self.model.eval()
        if isinstance(batch, Tensor):
            observed_variables: Tensor = batch.to(self.device)
        elif isinstance(batch, tuple) or isinstance(batch, list):
            observed_variables: Tensor = batch[0].to(self.device)
        else:
            raise ValueError(
                "batch must be either Tensor or tuple[Tensor]."
            )
        loss: Tensor = - self.model.calc_log_likelihood(
            observed_variables=observed_variables,
            reduction="joint",
            is_by_bit=True
        )
        recon_loss_: float = 0
        latent_variables, _ = self.model.backward(observed_variables)
        observed_variables_, _ = self.model(latent_variables)
        recon_loss_ = torch.sum(
            (observed_variables - observed_variables_).pow_(2)
        )
        if recon_loss is not None:
            recon_loss += recon_loss_
        valid_loss += float(loss) + self.recon_coef * recon_loss_
        return valid_loss, recon_loss

    def _save_checkpoint(self, epoch: int, seed: int):
        if self.valid_loss < self.best_loss:
            self.best_loss = self.valid_loss
            save_path = self.best_save_path
        else:
            save_path = self.last_save_path
        save_path = save_path.parent / f"{save_path.stem}_{seed}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "current_epoch": epoch + 1,
                "train_losses": self.train_losses,
                "train_recon_losses": self.train_recon_losses,
                "valid_losses": self.valid_losses,
                "valid_recon_losses": self.valid_recon_losses,
                "best_loss": self.best_loss
            },
            str(save_path)
        )

    def _create_dataloaders(
        self,
        batch_size: int = 256,
        num_workers: int = 2
    ) -> None:
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset, shuffle=True,
                batch_size=batch_size, num_workers=num_workers
            )
        if self.valid_dataset is not None:
            self.valid_dataloader = DataLoader(
                self.valid_dataset, shuffle=False,
                batch_size=batch_size, num_workers=num_workers
            )
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                self.test_dataset, shuffle=False,
                batch_size=batch_size, num_workers=num_workers
            )

    def _record_losses(self, epoch: int) -> None:
        self.train_losses.append(self.train_loss)
        self.valid_losses.append(self.valid_loss)
        if self.store_recon_loss:
            self.train_recon_losses.append(self.train_recon_loss)
            self.valid_recon_losses.append(self.valid_recon_loss)
        print_interval: int = int(self.end_epoch / 10)
        if epoch % print_interval == (print_interval-1):
            print(f"{epoch+1}/{self.end_epoch}epochs " +
                f"train_loss={self.train_loss:.4f} " +
                f"train_recon_loss={self.train_recon_loss:.7f}　" +
                f"valid_loss={self.valid_loss:.4f} " +
                f"valid_recon_loss={self.valid_recon_loss:.7f}"
            )

    def test(self):
        assert self.test_dataset is not None
        print()
        print(f"start testing. number of test samples: {self.test_n}")
        for seed in range(self.seed_range[0], self.seed_range[1]):
            if self.best_save_path is not None:
                load_path: Path = \
                    self.best_save_path.parent / \
                    f"{self.best_save_path.stem}_{seed}.pth"
                checkpoint = torch.load(
                    load_path,
                    map_location=torch.device(self.device)
                )
                self.model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=True
                )
            self._create_dataloaders()
            self.test_loss: float = 0.0
            self.test_recon_loss: float = 0.0
            for batch in self.test_dataloader:
                self.test_loss, self.test_recon_loss = self._validate_step(
                    batch, self.test_loss, self.test_recon_loss
                )
            self.test_loss = self.test_loss / self.test_n
            self.test_recon_loss = self.test_recon_loss / self.test_n
            print(
                f"seed: {seed} test_loss={self.test_loss:.4f} " +
                f"test_recon_loss={self.test_recon_loss:.7f}"
            )