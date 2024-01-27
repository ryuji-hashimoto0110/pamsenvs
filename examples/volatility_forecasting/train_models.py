import argparse
import json
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(parent_path))
from torch.nn import Module
from torch.nn import MSELoss
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from typing import Optional
from volatility_forecasting import RVDataset
from volatility_forecasting import RVPredictor
from volatility_forecasting import RVTrainer

datas_path: Path = root_path / "datas"
checkpoints_path: Path = curr_path / "checkpoints"
mean_std_dics_path: Path = curr_path / "mean_std_dics"
if not checkpoints_path.exists():
    checkpoints_path.mkdir(parents=True)
if not mean_std_dics_path.exists():
    mean_std_dics_path.mkdir(parents=True)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", type=str, required=True,
                        help="type of encoder. 'Transformer' or 'LSTM' is allowed.")
    parser.add_argument("--train_data_type", type=str, default=None,
                        help="type of train data type. 'artificial' or 'real' is allowed")
    parser.add_argument("--valid_data_type", type=str, default=None,
                        help="type of valid data type. 'artificial' or 'real' is allowed")
    parser.add_argument("--test_data_type", type=str, default=None,
                        help="type of test data type. 'artificial' or 'real' is allowed")
    parser.add_argument("--train_olhcv_name", type=str, default=None,
                        help="name of train olhcv folders.")
    parser.add_argument("--valid_olhcv_name", type=str, default=None,
                        help="name of valid olhcv folders.")
    parser.add_argument("--test_olhcv_name", type=str, default=None,
                        help="name of test olhcv folders.")
    parser.add_argument("--train_csv_names",
                        nargs="*", type=str, default=None,
                        help="names of csv files in train olhcv folder.")
    parser.add_argument("--valid_csv_names",
                        nargs="*", type=str, default=None,
                        help="names of csv files in valid olhcv folder.")
    parser.add_argument("--test_csv_names",
                        nargs="*", type=str, default=None,
                        help="names of csv files in test olhcv folder.")
    parser.add_argument("--train_obs_num", type=int, default=None,
                        help="observation number in 1 record of train dataset.")
    parser.add_argument("--valid_obs_num", type=int, default=None,
                        help="observation number in 1 record of valid dataset.")
    parser.add_argument("--test_obs_num", type=int, default=None,
                        help="observation number in 1 record of test dataset.")
    parser.add_argument("--train_mean_std_dic_name", type=str,
                        help="file name to load and save mean_std_dic training.")
    parser.add_argument("--test_mean_std_dic_name", type=str,
                        help="file name to load and save mean_std_dic for testing.")
    parser.add_argument("--criterion_type", type=str, default="MSE",
                        help="type of loss function. default to 'MSE'.")
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                        help="type of optimizer. default to 'AdamW'.")
    parser.add_argument("--learning_rate", type=float, default=3e-04,
                        help="learning rate. default to 0.0003")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of train epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch_size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers.")
    parser.add_argument("--load_name", type=str, default=None,
                        help="name of checkpoint file to be loaded.")
    parser.add_argument("--best_save_name", type=str, default=None,
                        help="name of best checkpoint file to be saved.")
    parser.add_argument("--last_save_name", type=str, default=None,
                        help="name of last checkpoint file to be saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed value. default to 42.")
    return parser

def create_dataset(
    data_type: str,
    olhcv_name: str,
    csv_names: Optional[list[str]],
    obs_num: int,
    input_time_length: int = 10,
    mean_std_dic: Optional[dict[str, dict[str, float]]] = None,
    mean_std_dic_save_path: Optional[Path] = None,
    imgs_path: Optional[Path] = None
) -> Dataset:
    assert olhcv_name is not None
    assert obs_num is not None
    assert data_type == "real" or data_type == "artificial"
    olhcv_path: Path = datas_path / f"{data_type}_datas" / "intraday" / olhcv_name
    rvdataset: Dataset = RVDataset(
        olhcv_path, csv_names, obs_num, input_time_length,
        mean_std_dic, mean_std_dic_save_path, imgs_path
    )
    return rvdataset


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    encoder_type: str = all_args.encoder_type
    model: Module = RVPredictor(
        encoder_type=encoder_type,
        input_dim=3, hidden_dim=512, nhead=8
    )
    train_data_type: Optional[str] = all_args.train_data_type
    valid_data_type: Optional[str] = all_args.valid_data_type
    test_data_type: Optional[str] = all_args.test_data_type
    train_olhcv_name: Optional[str] = all_args.train_olhcv_name
    valid_olhcv_name: Optional[str] = all_args.valid_olhcv_name
    test_olhcv_name: Optional[str] = all_args.test_olhcv_name
    train_csv_names: Optional[list[str]] = all_args.train_csv_names
    valid_csv_names: Optional[list[str]] = all_args.valid_csv_names
    test_csv_names: Optional[list[str]] = all_args.test_csv_names
    train_obs_num: Optional[int] = all_args.train_obs_num
    valid_obs_num: Optional[int] = all_args.valid_obs_num
    test_obs_num: Optional[int] = all_args.test_obs_num
    train_mean_std_dic_name: Optional[str] = all_args.train_mean_std_dic_name
    print(f"model type: {encoder_type}")
    print(f"data type: [tra]{train_data_type} [val]{valid_data_type} [test]{test_data_type}")
    print(f"olhcv folder name: [tra]{train_olhcv_name} [val]{valid_olhcv_name} [test]{test_olhcv_name}")
    print(f"use csv names: [tra]{train_csv_names} [val]{valid_csv_names} [test]{test_csv_names}")
    print(f"obs num per 1 record: [tra]{train_obs_num} [val]{valid_obs_num} [test]{test_obs_num}")
    mean_std_dic: Optional[dict[str, dict[str, float]]] = None
    if train_mean_std_dic_name is not None:
        train_mean_std_dic_path: Path = mean_std_dics_path / train_mean_std_dic_name
        if train_mean_std_dic_path.exists():
            mean_std_dic: dict[str, dict[str, float]] = json.load(
                fp=open(str(train_mean_std_dic_path), mode="r")
            )
    train_dataset: Optional[Dataset] = None
    valid_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    if train_data_type is not None:
        train_dataset: Dataset = create_dataset(
            data_type=train_data_type, olhcv_name=train_olhcv_name,
            csv_names=train_csv_names, obs_num=train_obs_num,
            mean_std_dic=mean_std_dic, mean_std_dic_save_path=train_mean_std_dic_path
        )
    if valid_data_type is not None:
        valid_dataset: Dataset = create_dataset(
            data_type=valid_data_type, olhcv_name=valid_olhcv_name,
            csv_names=valid_csv_names, obs_num=valid_obs_num,
            mean_std_dic=mean_std_dic
        )
    else:
        if train_dataset is not None:
            all_n: int = len(train_dataset)
            train_n: int = int(all_n * 0.7)
            train_dataset, valid_dataset = random_split(
                train_dataset, [train_n, all_n-train_n]
            )
    test_mean_std_dic_name: Optional[str] = all_args.test_mean_std_dic_name
    print(f"mean_std_dic: [tra]{train_mean_std_dic_name} [test]{test_mean_std_dic_name}")
    test_mean_std_dic: Optional[dict[str, dict[str, float]]] = None
    if test_mean_std_dic_name is not None:
        test_mean_std_dic_path: Path = mean_std_dics_path / test_mean_std_dic_name
        if test_mean_std_dic_path.exists():
            test_mean_std_dic: dict[str, dict[str, float]] = json.load(
                fp=open(str(test_mean_std_dic_path), mode="r")
            )
    if test_data_type is not None:
        assert test_mean_std_dic is not None
        test_dataset: Dataset = create_dataset(
            data_type=test_data_type, olhcv_name=test_olhcv_name,
            csv_names=test_csv_names, obs_num=test_obs_num,
            mean_std_dic=test_mean_std_dic
        )
    criterion_type: str = all_args.criterion_type
    if criterion_type == "MSE":
        criterion: Module = MSELoss(reduction="sum")
    else:
        raise NotImplementedError
    optimizer_type: str = all_args.optimizer_type
    lr: float = all_args.learning_rate
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    else:
        raise NotImplementedError
    print(f"criterion: {criterion_type} optimizer: {optimizer_type}")
    num_epochs: int = all_args.num_epochs
    batch_size: int = all_args.batch_size
    num_workers: int = all_args.num_workers
    load_name: str = all_args.load_name
    best_save_name: str = all_args.best_save_name
    last_save_name: str = all_args.last_save_name
    print(f"num epochs: {num_epochs} batch size: {batch_size} num_workers: {num_workers}")
    print(f"load name: {load_name} save name: [best]{best_save_name} [last]{last_save_name}")
    if load_name is not None:
        load_path: Path = checkpoints_path / load_name
    else:
        load_path = None
    if best_save_name is not None:
        best_save_path: Path = checkpoints_path / best_save_name
    else:
        best_save_path = None
    if last_save_name is not None:
        last_save_path: Path = checkpoints_path / last_save_name
    else:
        best_save_path = None
    seed: int = all_args.seed
    print(f"seed={seed}")
    trainer = RVTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        train_dataset=train_dataset, valid_dataset=valid_dataset,
        test_dataset=test_dataset, num_epochs=num_epochs,
        batch_size=batch_size, num_workers=num_workers,
        load_path=load_path,
        best_save_path=best_save_path, last_save_path=last_save_path,
        seed=seed
    )
    if train_dataset is not None:
        trainer.fit()
    if test_dataset is not None:
        trainer.test()

if __name__ == "__main__":
    main(sys.argv[1:])