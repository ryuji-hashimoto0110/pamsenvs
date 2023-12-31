from matplotlib.pyplot import Axes
import numpy as np
from numpy import ndarray
from pams.logs import Logger
from pams.logs import MarketStepEndLog
from pams.market import Market
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, TypeVar

MarketID = TypeVar("MarketID")

class VolumePriceSaver(Logger):
    """Saver of the market step class.

    This is different from pams.logs.MarketStepSaver in that this also saves
    execution volumes.

    VolumePriceSaver.logs_dic has following lists.
        - times (list[int])
        - market_prices (list[float])
        - fundamental_prices (list[float])
        - execution_volumes (list[int])
    """
    def __init__(self) -> None:
        super().__init__()
        self.logs_dic: dict[MarketID, dict[str, list]] = {}
        self.times: list[int] = []
        self.market_prices: list[float] = []
        self.fundamental_prices: list[float] = []
        self.execution_volumes: list[int] = []

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """stack the market log."""
        market: Market = log.market
        market_id: MarketID = market.market_id
        if market_id not in list(self.logs_dic.keys()):
            self.logs_dic[market_id] = {
                "times": [],
                "market_prices": [],
                "fundamental_prices": [],
                "execution_volumes": []
            }
        self._add_log_by_market(market_id, log)

    def _add_log_by_market(
        self,
        market_id: MarketID,
        log: MarketStepEndLog
    ) -> None:
        market: Market = log.market
        self.logs_dic[market_id]["times"].append(market.get_time())
        self.logs_dic[market_id]["market_prices"].append(market.get_market_price())
        self.logs_dic[market_id]["fundamental_prices"].append(market.get_fundamental_price())
        self.logs_dic[market_id]["execution_volumes"].append(market.get_executed_volume())

    def plot_volume_prices(
        self,
        ax: Axes,
        market_id: MarketID,
        time_interval: Optional[list[int]] = None
    ) -> None:
        logs_dic: dict[str, list] = self.logs_dic[market_id]
        indices: list[int] = self._get_time_indices(logs_dic["times"], time_interval)
        times: list[int] = logs_dic["times"][indices[0]:indices[1]]
        market_prices: list[float] = logs_dic["market_prices"][indices[0]:indices[1]]
        fundamental_prices: list[float] = logs_dic["fundamental_prices"][indices[0]:indices[1]]
        execution_volumes: list[int] = logs_dic["execution_volumes"][indices[0]:indices[1]]
        ax.plot(times, fundamental_prices, color="red", label="fundamental price")
        ax.plot(times, market_prices, color="black", label="market price")
        ax_: Axes = ax.twinx()
        ax_.bar(times, execution_volumes, align="center", width=1.0,
                color="green", label="execution volume")
        lines, labels = ax.get_legend_handles_labels()
        lines_, labels_ = ax_.get_legend_handles_labels()
        ax.legend(lines+lines_, labels+labels_)

    def _get_time_indices(
        self,
        times: list[int],
        time_interval: Optional[list[int]] = None,
    ) -> list[int]:
        if time_interval is None:
            return [0, len(times)]
        assert len(time_interval) == 2
        indices: list[int] = [
            times.index(time_interval[0]),
            times.index(time_interval[1])
        ]
        return indices

    def save_olhcv(
        self,
        market_id: MarketID,
        start_index: int,
        index_interval: int,
        save_path: Path
    ):
        logs_dic: dict[str, list] = self.logs_dic[market_id]
        times_arr: ndarray = np.array(logs_dic["times"], dtype=np.uint8)[start_index::index_interval]
        prices_arr: ndarray = np.array(logs_dic["market_prices"], dtype=np.float32)[start_index:]
        volumes_arr: ndarray = np.array(logs_dic["execution_volumes"], dtype=np.uint8)[start_index:]
        prices_arr = self._reshape2matrix(prices_arr, index_interval)
        volumes_arr = self._reshape2matrix(volumes_arr, index_interval)

    def _back_pad(
        self,
        arr: ndarray,
        len_padded_arr: int
    ) -> ndarray:
        assert len(arr) < len_padded_arr
        if len(arr) == len_padded_arr:
            return arr
        padded_arr: ndarray = np.empty(len_padded_arr, dtype=type(arr))
        padded_arr[:len(arr)] = arr
        padded_arr[len(arr):] = arr[-1]
        return padded_arr

    def _reshape2matrix(
        self,
        arr: ndarray,
        index_interval: int
    ):
        row_num: int = int(np.ceil(len(arr) / index_interval) * index_interval)
        padded_arr: ndarray = self._back_pad(arr, row_num)
        reshaped_arr: ndarray = padded_arr.reshape(row_num, index_interval)
        return reshaped_arr

