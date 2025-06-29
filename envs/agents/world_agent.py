
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
        num_channels: int,
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


class ConditionalTimeSeriesGenerator(Module):
    def __init__(
        self,
        history_len: int,
        hist_feat_dim: int,
        next_feat_dim: int,
        noise_dim: int = 100,
        tcn_channels: tuple[int, int] = (128, 128),
        mlp_channels: int = 256
    ) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(
            hist_feat_dim, tcn_channels, kernel_size=3, dropout=0.1
        )
        self.noise_fc = nn.Linear(noise_dim, tcn_channels[-1])
        self.mlp = nn.Sequential(
            nn.Linear(tcn_channels[-1] * 2, mlp_channels),
            nn.ReLU(True),
            nn.Linear(mlp_channels, next_feat_dim),
            nn.Tanh()
        )

    def forward(self, noise: Tensor, history: Tensor) -> Tensor:
        hist_emb = self.tcn(history)
        noise_emb = self.noise_fc(noise)
        out = self.mlp(torch.cat([hist_emb, noise_emb], dim=1))
        return out


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
        self.condition_len: int = 0
        self.noise_dim: int = 0
        self.device: Optional[torch.device] = None
        self.condition_columns: list[str] = [
            "price", "size", "side",
            "best_ask_price", "best_ask_volume",
            "best_bid_price", "best_bid_volume", "mid_price"
        ]
        self.order_columns: list[str] = ["price", "size"]
        self.box_cox_cols: list[str] = [
            #"size", "best_ask_volume", "best_bid_volume"
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
        self.condition_len: int = settings["condition_len"] \
            if "condition_len" in settings else 50
        self.noise_dim: int = settings["noise_dim"] \
            if "noise_dim" in settings else 50
        self.device: torch.device = torch.device(settings["device"]) \
            if "device" in settings else torch.device("cpu")
        self.generator = ConditionalTimeSeriesGenerator(
            history_len=self.condition_len,
            hist_feat_dim=len(self.condition_columns),
            next_feat_dim=len(self.order_columns),
            noise_dim=self.noise_dim
        ).to(self.device)
        if "generator_weight_path" not in settings:
            warnings.warn(
                "No 'generator_weight_path' found in settings. "
                "WorldAgent might not be initialized properly."
            )
        else:
            generator_weight_path: Path = self._load_path(
                settings, "generator_weight_path"
            )
            checkpoint = torch.load(generator_weight_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['model'])
            self.generator.eval()
        boxcox_lambdas_path: Path = self._load_path(
            settings, "boxcox_lambdas_path"
        )
        with open(boxcox_lambdas_path, 'rb') as f:
            self.boxcox_lambdas: dict[str, float] = pickle.load(f)

    def _get_market_history(self, market: Market) -> DataFrame:
        if hasattr(market, "order_history_df"):
            order_history_df: DataFrame = market.order_history_df
        else:
            raise ValueError(
                f"Market {market.market_id} does not have 'order_history_df' attribute."
            )
        if len(order_history_df) < self.condition_len:
            raise ValueError(
                f"Market {market.market_id} does not have enough history data. "
                f"Required: {self.condition_len}, Available: {len(order_history_df)}"
            )
        return order_history_df.tail(self.condition_len).reset_index(drop=True)

    def _convert_df2tensor(self, df: DataFrame) -> Tensor:
        assert len(df) == self.condition_len
        preprocessed_df: DataFrame = df.copy()
        for col in self.box_cox_cols:
            if preprocessed_df[col].min() <= 0:
                preprocessed_df[col] += abs(preprocessed_df[col].min()) + 1e-6
            preprocessed_df[col] = (
                preprocessed_df[col]**self.boxcox_lambdas[col] - 1
            ) / self.boxcox_lambdas[col]
        input_tensor: Tensor = torch.tensor(
            preprocessed_df[self.condition_columns].values,
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        assert input_tensor.shape == (1, self.condition_len, len(self.condition_columns))
        return input_tensor
    
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
        if len(market.order_history_df) < self.condition_len:
            return []
        history_df: DataFrame = self._get_market_history(market)
        history_tensor: Tensor = self._convert_df2tensor(history_df)
        noise_tensor: Tensor = torch.randn(
            (1, self.noise_dim), device=self.device
        )
        with torch.no_grad():
            order_tensor: Tensor = self.generator(
                noise_tensor, history_tensor
            )
        orders: list[Order] = self._convert_tensor2orders(
            order_tensor, market.market_id
        )
        return orders