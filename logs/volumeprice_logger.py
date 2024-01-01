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
        """add market log to logs_dic["market_id"].
        """
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
        """plot prices and volumes.

        set time_interval to draw specific time interval. Ex: time_interval=[100,2000]
        """
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
        """get time indices from given time interval.
        """
        if time_interval is None:
            return [0, len(times)]
        assert len(time_interval) == 2
        assert time_interval[0] in times
        assert time_interval[1] in times
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
    ) -> None:
        """save OLHCV csv data.

        Args:
            market_id (MarketID): market id.
            start_index (int): first index to tally the data.
            index_interval (int): fixed time interval to tally OLHCV.
            save_path (Path): csv save path.
        """
        logs_dic: dict[str, list] = self.logs_dic[market_id]
        times_arr: ndarray = np.array(
            logs_dic["times"]
        )[start_index::index_interval]
        prices_arr: ndarray = np.array(
            logs_dic["market_prices"]
        )[start_index:]
        volumes_arr: ndarray = np.array(
            logs_dic["execution_volumes"]
        )[start_index:]
        prices_arr = self._reshape2matrix(prices_arr, index_interval)
        volumes_arr = self._reshape2matrix(volumes_arr, index_interval, 0)
        olhcv_df: DataFrame = pd.DataFrame(
            {
                "open": prices_arr[:,0],
                "low": np.min(prices_arr, axis=1),
                "high": np.max(prices_arr, axis=1),
                "close": prices_arr[:,-1],
                "volume": np.sum(volumes_arr, axis=1)
            },
            index=times_arr
        )
        if save_path.suffix != "csv":
            save_path.with_suffix(".csv")
        olhcv_df.to_csv(str(save_path))

    def _back_pad(
        self,
        arr: ndarray,
        len_padded_arr: int,
        padding_num: Optional[float] = None
    ) -> ndarray:
        """apply back padding to arr

        back pad arr to length len_padded_arr. pad with the number padding_num.
        """
        assert len(arr.shape) == 1
        assert len(arr) < len_padded_arr
        if len(arr) == len_padded_arr:
            return arr
        padded_arr: ndarray = np.empty(len_padded_arr, dtype=type(arr))
        padded_arr[:len(arr)] = arr
        if padding_num is None:
            padded_arr[len(arr):] = arr[-1]
        else:
            padded_arr[len(arr):] = padding_num
        return padded_arr

    def _reshape2matrix(
        self,
        arr: ndarray,
        index_interval: int,
        padding_num: Optional[float] = None
    ) -> ndarray:
        """reshape 1 dim numpy vector to matrix whose number of columns is index_interval.

        Ex) arr = [1,2,3,4,5,6], index_interval = 3
            -> [[1,2,3],
                [4,5,6]]

        If arr cannot reshape to matrix, back padding is applied.

        Ex) arr = [1,2,3,4,5], index_interval = 3, padding_num = 0
            -> [[1,2,3],
                [4,5,0]]

        Args:
            arr (ndarray): 1 dim numpy vector.
            index_interval (int): fixed time interval
            padding_num (Optional[float]): number used to pad. default to None.

        Returns:
            reshaped_arr: reshaped matrix.
        """
        assert len(arr.shape) == 1
        row_num: int = int(np.ceil(len(arr) / index_interval))
        len_padded_arr: int = row_num * index_interval
        if len(arr) < len_padded_arr:
            arr: ndarray = self._back_pad(arr, len_padded_arr, padding_num)
        reshaped_arr: ndarray = arr.reshape(row_num, index_interval)
        return reshaped_arr

