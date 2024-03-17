import argparse
from numpy import ndarray
import pathlib
from pathlib import Path
import sys
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[0]
sys.path.append(str(root_path))
from flows import CircleDataset2d
from flows import get_config
from flows import FlowModel
from flows import FlowTrainer
from flows import PlanarFlow
from flows import RealNVP
import json
from rich import print
from rich.tree import Tree
from rich.text import Text
import sys
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets
import torchvision.transforms as transforms
from typing import Literal
from typing import Optional

def parse_args(args, parser):
    parser.add_argument(
        "--recon_coef", type=float
    )
    parser.add_argument(
        "--optimizer_type", type=str, choices=["Adam"]
    )
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument(
        "--checkpoints_name", type=str
    )
    parser.add_argument(
        "--store_recon_loss", type=str, default="false"
    )
    parser.add_argument(
        "--load_name", type=str, default=None
    )
    parser.add_argument(
        "--best_save_name", type=str, default=None
    )
    parser.add_argument(
        "--last_save_name", type=str, default=None
    )
    parser.add_argument(
        "--train_folder_name", type=str, default=None
    )
    parser.add_argument(
        "--test_folder_name", type=str, default=None
    )
    parser.add_argument(
        "--radius", type=float, default=1.0
    )
    parser.add_argument(
        "--center", type=int, nargs="+"
    )
    parser.add_argument(
        "--randn_std", type=float, default=0.1
    )
    parser.add_argument(
        "--num_samples", type=int, nargs="+"
    )
    parser.add_argument("--seed_range", type=int, nargs="+",
                    default=[42, 43])
    all_args = parser.parse_known_args(args)[0]
    return all_args

model_dic: dict[str, FlowModel] = {
    "planar": PlanarFlow,
    "realnvp": RealNVP
}

def create_model(
    flow_type: str,
    config_dic: dict[str, int]
) -> FlowModel:
    print("[red]==create model==[/red]")
    if flow_type not in model_dic.keys():
        raise NotImplementedError
    print(
        f"flow type: {flow_type}",
        config_dic
    )
    print()
    model: FlowModel = model_dic[flow_type](config_dic)
    return model

def create_circle2d_dataset(
    radius: float,
    center: list[float] | ndarray,
    randn_std: float,
    num_samples: list[int] = [10000, 1000, 1000]
) -> tuple[Dataset]:
    print("[red]==create circle 2d datasets==[/red]")
    train_dataset: Dataset = CircleDataset2d(
        radius, center, randn_std, num_samples[0]
    )
    valid_dataset: Dataset = CircleDataset2d(
        radius, center, randn_std, num_samples[1]
    )
    test_dataset: Dataset = CircleDataset2d(
        radius, center, randn_std, num_samples[2]
    )
    print(
        f"radius: {radius} center: {center} " +
        f"randn_std: {randn_std}"
    )
    print()
    return train_dataset, valid_dataset, test_dataset

def create_image_dataset(
    train_path: Path,
    test_path: Path,
    data_name: Literal["mnist", "cifar10"]
) -> tuple[Dataset]:
    print("[red]==create MNIST datasets==[/red]")
    if data_name == "mnist":
        trainval_dataset: Dataset = datasets.MNIST(
            train_path, train=True, download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5],[0.55]
                    )
                ]
            )
        )
        test_dataset: Dataset = datasets.MNIST(
            test_path, train=False, download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5],[0.55]
                    )
                ]
            )
        )
    elif data_name == "cifar10":
        trainval_dataset: Dataset = datasets.CIFAR10(
            train_path, train=True, download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5],[0.55]
                    )
                ]
            )
        )
        test_dataset: Dataset = datasets.CIFAR10(
            test_path, train=False, download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5],[0.55]
                    )
                ]
            )
        )
    all_n: int = len(trainval_dataset)
    train_n: int = int(all_n * 0.7)
    train_dataset, valid_dataset = random_split(
        trainval_dataset, [train_n, all_n-train_n]
    )
    tree = Tree(str(train_path.parent.resolve()))
    tree.add(train_path.name)
    tree.add(test_path.name)
    print(tree)
    print()
    return train_dataset, valid_dataset, test_dataset

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    flow_type: str = all_args.flow_type
    config_json_name: str = all_args.config_json_name
    config_path: Path = root_path / config_json_name
    if config_path.exists():
        config_dic: dict[str, int] = json.load(
            fp=open(str(config_path), mode="r")
        )
    else:
        raise ValueError(
            f"config_path: {str(config_path.resolve())} does not exist."
        )
    model: FlowModel = create_model(flow_type, config_dic)
    data_type: str = all_args.data_type
    if data_type == "mnist" or data_type == "cifar10":
        train_folder_name: str = all_args.train_folder_name
        test_folder_name: str = all_args.test_folder_name
        train_path: Path = root_path / train_folder_name
        test_path: Path = root_path / test_folder_name
        train_dataset, valid_dataset, test_dataset = create_image_dataset(
            train_path, test_path, data_type
        )
    elif data_type == "circle2d":
        radius: float = all_args.radius
        center: list[float] = all_args.center
        randn_std: float = all_args.randn_std
        num_samples: list[int] = all_args.num_samples
        train_dataset, valid_dataset, test_dataset = create_circle2d_dataset(
            radius, center, randn_std, num_samples
        )
    else:
        raise NotImplementedError
    print("[red]==trainer settings==[/red]")
    recon_coef: float = all_args.recon_coef
    optimizer_type: str = all_args.optimizer_type
    lr: float = all_args.learning_rate
    weight_decay: float = all_args.weight_decay
    if optimizer_type != "Adam":
        raise NotImplementedError
    num_epochs: int = all_args.num_epochs
    batch_size: int = all_args.batch_size
    num_workers: int = all_args.num_workers
    seed_range: list[int] = all_args.seed_range
    store_recon_loss: str = all_args.store_recon_loss
    if (
        store_recon_loss == "True" or
        store_recon_loss == "true" or
        store_recon_loss == "TRUE"
    ):
        store_recon_loss: bool = True
    else:
        store_recon_loss: bool = False
    checkpoints_name: str = all_args.checkpoints_name
    load_name: str = all_args.load_name
    best_save_name: str = all_args.best_save_name
    last_save_name: str = all_args.last_save_name
    checkpoints_path: Path = root_path / checkpoints_name
    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)
    tree = Tree(str(checkpoints_path.resolve()))
    load_path: Optional[Path] = None
    best_save_path: Optional[Path] = None
    last_save_path: Optional[Path] = None
    if load_name is not None:
        load_path: Path = checkpoints_path / load_name
        tree.add(f"load path: {load_path.name}")
    if best_save_name is not None:
        best_save_path: Path = checkpoints_path / best_save_name
        tree.add(f"best save path: {best_save_path.name}")
    if last_save_name is not None:
        last_save_path: Path = checkpoints_path / last_save_name
        tree.add(f"last save path: {last_save_path.name}")
    print(f"optimizer: {optimizer_type}")
    print(
        f"coefficient of reconstruction loss: {recon_coef:.4f} " +
        f"learning rate: {lr} number of epochs: {num_epochs} " +
        f"batch size: {batch_size} number of workers: {num_workers} " +
        f"store reconstruction loss: {store_recon_loss}"
    )
    print(f"seed range: {seed_range}")
    print(f"checkpoints:", tree)
    print()
    trainer: FlowTrainer = FlowTrainer(
        model, recon_coef, optimizer_type, lr, weight_decay,
        train_dataset, valid_dataset, test_dataset,
        num_epochs, batch_size, num_workers, store_recon_loss,
        load_path, best_save_path, last_save_path,
        seed_range
    )
    if train_dataset is not None:
        trainer.fit()
    if test_dataset is not None:
        trainer.test()

if __name__ == "__main__":
    main(sys.argv[1:])