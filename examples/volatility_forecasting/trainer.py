import numpy as np
from pathlib import Path
import random
import torch
from torch import Tensor
from torch.nn import Module
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional

class RVTrainer:
    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        train_dataset: Optional[Dataset],
        valid_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        num_epochs: int,
        batch_size: int,
        num_workers: int,
        load_path: Optional[Path] = None,
        best_save_path: Optional[Path] = None,
        last_save_path: Optional[Path] = None,
        seed: int = 42
    ) -> None:
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
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
        self.best_save_path: Optional[Path] = best_save_path
        self.last_save_path: Optional[Path] = last_save_path
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    def fit(self) -> None:
        print(f"start training. train_n{self.train_n} valid_n{self.valid_n}")
        self._setup()
        assert self.train_dataset is not None
        assert self.valid_dataset is not None
        self._create_dataloaders(self.batch_size, self.num_workers)
        for epoch in range(self.start_epoch, self.end_epoch):
            self.train_loss: float = 0.0
            for batch in self.train_dataloader:
                self._train_step(batch)
            self.valid_loss: float = 0.0
            for batch in self.valid_dataloader:
                self.valid_loss = self._validate_step(batch, self.valid_loss)
            self.train_loss = np.sqrt(self.train_loss / self.train_n)
            self.valid_loss = np.sqrt(self.valid_loss / self.valid_n)
            self._record_losses(epoch)
            self._save_checkpoint(epoch)

    def _train_step(
        self,
        batch: tuple[Tensor, Tensor]
    ) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        inputs: Tensor = batch[0].to(self.device)
        targets: Tensor = batch[1].to(self.device)
        outputs: Tensor = self.model(inputs)
        loss: Tensor = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.train_loss += float(loss)

    @torch.no_grad()
    def _validate_step(
        self,
        batch: tuple[Tensor, Tensor],
        valid_loss: float
    ) -> float:
        self.model.eval()
        inputs: Tensor = batch[0].to(self.device)
        targets: Tensor = batch[1].to(self.device)
        outputs: Tensor = self.model(inputs)
        loss: Tensor = self.criterion(outputs, targets)
        valid_loss += float(loss)
        return valid_loss

    def _save_checkpoint(self, epoch: int):
        if self.valid_loss < self.best_loss:
            self.best_loss = self.valid_loss
            save_path = self.best_save_path
        else:
            save_path = self.last_save_path
        if save_path is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "current_epoch": epoch + 1,
                    "train_losses": self.train_losses,
                    "valid_losses": self.valid_losses,
                    "best_loss": self.best_loss
                },
                str(save_path)
            )
            print(f"model saved >> {str(save_path)}")

    def _setup(self) -> None:
        if self.load_path is not None:
            checkpoint = torch.load(
                self.load_path, map_location=torch.device(self.device)
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.start_epoch: int = checkpoint["current_epoch"]
            self.end_epoch: int = self.start_epoch + self.num_epochs
            self.train_losses: list[float] = checkpoint["train_losses"]
            self.valid_losses: list[float] = checkpoint["valid_losses"]
            self.best_loss: list[float] = checkpoint["best_loss"]
        else:
            self.start_epoch: int = 0
            self.end_epoch: int = self.num_epochs
            self.train_losses: list[float] = []
            self.valid_losses: list[float] = []
            self.best_loss: float = 1e+10

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
        print(f"{epoch+1}/{self.end_epoch}epochs " +
            f"loss[tra]{self.train_loss:.4f} [val]{self.valid_loss:.4f}")

    def test(self):
        assert self.test_dataset is not None
        print(f"start testing. test_n={self.test_n}")
        if self.best_save_path is not None:
            checkpoint = torch.load(
                self.best_save_path, map_location=torch.device(self.device)
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._create_dataloaders()
        self.test_loss: float = 0.0
        for batch in self.test_dataloader:
            self.test_loss = self._validate_step(batch, self.test_loss)
        self.test_loss = np.sqrt(self.test_loss / self.test_n)
        print(f"loss[test]{self.test_loss:.4f}")