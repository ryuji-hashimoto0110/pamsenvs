
import numpy as np
from numpy import ndarray
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Order
from pams.order import LIMIT_ORDER
from pams.simulator import Simulator
import pandas as pd
from pandas import DataFrame
import pathlib
from pathlib import Path
import pickle
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Any
from typing import Optional
import warnings


class Simlog(Module):
    def __init__(self) -> None:
        super(Simlog, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1)


class TemporalBlock(Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=0, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            stride=stride, padding=0, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = F.pad(x, (self.padding, 0))
        out: Tensor = self.relu(self.bn1(self.conv1(out)))
        out: Tensor = self.dropout1(out)
        out: Tensor = F.pad(out, (self.padding, 0))
        out: Tensor = self.relu(self.bn2(self.conv2(out)))
        out: Tensor = self.dropout2(out)
        res: Tensor = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: tuple[int, ...],
        kernel_size: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[Module] = []
        for i in range(len(num_channels)):
            in_ch: int = num_inputs if i == 0 else num_channels[i-1]
            out_ch: int = num_channels[i]
            dilation: int = 2**i
            padding: int = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    dilation=dilation, padding=padding, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.network(x.transpose(1, 2))
        return out[:, :, -1]


class HistoryAwareTimeSeriesGenerator(Module):
    def __init__(
        self,
        len_order_history: int,
        dim_order_history_features: int,
        dim_noise: int = 50,
        dim_noise_embedding: int = 64,
        dim_condition: int = 1,
        dim_condition_embedding: int = 64,
        dim_order: int = 2
    ) -> None:
        super().__init__()
        self.tcn: TemporalConvNet = TemporalConvNet(
            len_order_history, (128, 128), kernel_size=3, dropout=0.1
        )
        self.noise_fc: Module = nn.Linear(dim_noise, dim_noise_embedding)
        self.condition_mlp: Optional[Module] = None
        if 0 < dim_condition:
            self.condition_mlp: Module = nn.Sequential(
                nn.Linear(dim_condition, dim_condition_embedding),
                nn.LeakyReLU(0.2, True),
                nn.Linear(dim_condition_embedding, dim_condition_embedding),
            )
        else:
            dim_condition_embedding = 0
        self.mlp = nn.Sequential(
            nn.Linear(128+dim_noise_embedding+dim_condition_embedding, 256),
            nn.ReLU(True),
            nn.Linear(256, dim_order),
            Simlog(),
        )

    def forward(
        self,
        noise_tensor: Tensor,
        order_history_tensor: Tensor,
        condition_tensor: Optional[Tensor] = None
    ) -> Tensor:
        noise_emb_tensor: Tensor = self.noise_fc(noise_tensor)
        order_history_emb: Tensor = self.tcn(order_history_tensor)
        if condition_tensor is not None:
            assert self.condition_mlp is not None
            condition_emb_tensor: Tensor = self.condition_mlp(condition_tensor)
            order_tensor: Tensor = self.mlp(
                torch.cat(
                    [noise_emb_tensor, order_history_emb, condition_emb_tensor], dim=1
                )
            )
        else:
            order_tensor: Tensor = self.mlp(torch.cat([noise_emb_tensor, order_history_emb], dim=1))
        return order_tensor
    

class DummyAgent(Agent):
    def submit_orders(self, markets: list[Market]) -> list[Order]:
        return []


class WorldAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(agent_id, prng, simulator, name, logger)
        self.generator: Optional[Module] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.boxcox_lambdas: Optional[dict[str, float]] = None
        self.device: Optional[torch.device] = None
        self.order_history_columns: list[str] = [
            "order_price", "order_size",
            "best_ask_price", "best_ask_volume",
            "best_bid_price", "best_bid_volume", "mid_price"
        ]
        self.order_columns: list[str] = ["order_price", "order_size"]
        self.box_cox_cols: list[str] = [
            "order_size", "best_ask_volume", "best_bid_volume"
        ]

    def _load_path(
        self,
        settings: dict[str, Any],
        path_key: str
    ) -> Path:
        if path_key not in settings:
            raise ValueError(f"'{path_key}' not found in settings.")
        path_str: str = settings[path_key]
        path: Path = pathlib.Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path

    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[int]
    ) -> None:
        super().setup(settings, accessible_markets_ids)
        if "dimOrderHistoryFeatures" not in settings:
            raise ValueError(
                "Specify 'dimOrderHistoryFeatures' in settings."
            )
        self.dim_order_history_features: int = settings["dimOrderHistoryFeatures"]
        self.len_order_history: int = settings["lenOrderHistory"] \
            if "lenOrderHistory" in settings else 50
        self.noise_dim: int = settings["dimNoise"] \
            if "dimNoise" in settings else 50
        self.dim_condition: int = settings["dimCondition"] \
            if "dimCondition" in settings else 0
        self.conditions: list[float] = settings["condition"] \
            if "condition" in settings else []
        assert isinstance(self.conditions, list)
        assert len(self.conditions) == self.dim_condition
        self.device: torch.device = torch.device(settings["device"]) \
            if "device" in settings else torch.device("cpu")
        self.generator = HistoryAwareTimeSeriesGenerator(
            len_order_history=self.len_order_history,
            dim_order_history_features=self.dim_order_history_features,
            dim_noise=self.noise_dim,
            dim_condition=self.dim_condition,
        ).to(self.device)
        if "generatorWeightPath" not in settings:
            warnings.warn(
                "No 'generatorWeightPath' found in settings. "
                "WorldAgent might not be initialized properly."
            )
        else:
            generator_weight_path: Path = self._load_path(
                settings, "generatorWeightPath"
            )
            checkpoint = torch.load(generator_weight_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['model'])
            self.generator.eval()
        if "boxcoxLambdasPath" not in settings:
            warnings.warn(
                "No 'boxcoxLambdasPath' found in settings. "
                "WorldAgent might not be initialized properly."
            )
        else:
            boxcox_lambdas_path: Path = self._load_path(
                settings, "boxcoxLambdasPath"
            )
            with open(boxcox_lambdas_path, 'rb') as f:
                self.boxcox_lambdas: dict[str, float] = pickle.load(f)
                for col in self.box_cox_cols:
                    if col not in self.boxcox_lambdas:
                        raise ValueError(
                            f"Column '{col}' not found in boxcox_lambdas."
                        )

    def _get_market_history(self, market: Market) -> DataFrame:
        if hasattr(market, "order_history_df"):
            order_history_df: DataFrame = market.order_history_df
        else:
            raise ValueError(
                f"Market {market.market_id} does not have 'order_history_df' attribute."
            )
        if len(order_history_df) < self.len_order_history + 1:
            raise ValueError(
                f"Market {market.market_id} does not have enough history data. "
                f"Required: {self.len_order_history}, Available: {len(order_history_df)}"
            )
        return order_history_df.tail(self.len_order_history+1).reset_index(drop=True)

    def _convert_df2tensor(self, df: DataFrame) -> Tensor:
        assert len(df) == self.len_order_history + 1
        preprocessed_df: DataFrame = df.copy()
        preprocessed_df = self._boxcox_transform(preprocessed_df)
        order_price_tensor: Tensor = self._get_broad_log_return_tensor(
            preprocessed_df, "order_price"
        )
        order_size_tensor: Tensor = torch.tensor(
            preprocessed_df["order_size"].values[1:], dtype=torch.float32, device=self.device
        )
        best_ask_price_tensor: Tensor = self._get_broad_log_return_tensor(
            preprocessed_df, "best_ask_price"
        )
        best_ask_volume_tensor: Tensor = torch.tensor(
            preprocessed_df["best_ask_volume"].values[1:], dtype=torch.float32, device=self.device
        )
        best_bid_price_tensor: Tensor = self._get_broad_log_return_tensor(
            preprocessed_df, "best_bid_price"
        )
        best_bid_volume_tensor: Tensor = torch.tensor(
            preprocessed_df["best_bid_volume"].values[1:], dtype=torch.float32, device=self.device
        )
        mid_price_tensor: Tensor = self._get_broad_log_return_tensor(
            preprocessed_df, "mid_price"
        )
        input_tensor: Tensor = torch.stack(
            [
                order_price_tensor,
                order_size_tensor,
                best_ask_price_tensor,
                best_ask_volume_tensor,
                best_bid_price_tensor,
                best_bid_volume_tensor,
                mid_price_tensor
            ],
            dim=0
        ).unsqueeze(0).to(self.device)
        assert input_tensor.shape == (1, self.len_order_history, self.dim_order_history_features)
        return input_tensor
    
    def _boxcox_transform(self, df: DataFrame) -> DataFrame:
        for col in self.box_cox_cols:
            if df[col].min() <= 0:
                df[col] += abs(df[col].min()) + 1e-6
            df[col] = (
                df[col]**self.boxcox_lambdas[col] - 1
            ) / self.boxcox_lambdas[col]
        return df
    
    def _get_broad_log_return_tensor(
        self,
        df: DataFrame,
        col_name: str
    ) -> Tensor:
        log_return_arr: ndarray = np.log(df[col_name].values[1:] / df[col_name].values[0])
        log_return_tensor: Tensor = torch.tensor(
            log_return_arr, dtype=torch.float32, device=self.device
        )
        return log_return_tensor 

    def _postprocess_order_price(self, order_price: float) -> float:
        return order_price
    
    def _postprocess_order_size(self, order_size: float) -> int:
        order_size: int = int(order_size)
        return order_size

    def _convert_tensor2orders(
        self,
        output_tensor: Tensor,
        market_id: int
    ) -> list[Order]:
        output_arr: ndarray = output_tensor.squeeze(0).detach().cpu().numpy()
        assert output_arr.shape == (len(self.order_columns),)
        order_price: float = output_arr[0]
        order_price = self._postprocess_order_price(order_price)
        order_size: float = output_arr[1]
        order_size: int = self._postprocess_order_size(order_size)
        if order_size == 0:
            return []
        is_buy: int = True if order_size >= 0 else False
        return [
            Order(
                agent_id=self.agent_id,
                market_id=market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=abs(order_size),
                ttl=100,
                price=order_price,
            )
        ]

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order]:
        if 1 < len(markets):
            warnings.warn(
                "WorldAgent currently supports only one market. "
                "Using the first market in the list."
            )
        market: Market = markets[0]
        assert hasattr(market, "order_history_df")
        if len(market.order_history_df) < self.len_order_history + 1:
            return []
        history_df: DataFrame = self._get_market_history(market)
        history_tensor: Tensor = self._convert_df2tensor(history_df)
        noise_tensor: Tensor = torch.randn(
            (1, self.noise_dim), device=self.device
        )
        condition_tensor: Optional[Tensor] = None
        if self.dim_condition > 0:
            condition_tensor: Tensor = torch.tensor(
                self.conditions, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        with torch.no_grad():
            order_tensor: Tensor = self.generator(
                noise_tensor, history_tensor, condition_tensor
            )
        orders: list[Order] = self._convert_tensor2orders(
            order_tensor, market.market_id
        )
        return orders