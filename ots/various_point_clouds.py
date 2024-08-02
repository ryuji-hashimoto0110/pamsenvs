from .data_distance_evaluater import DDEvaluater
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional

freq_ohlcv_size_dic: dict[str, int] = {
    "1s": 18001,
    "10s": 1801,
    "30s": 601,
    "1min": 301,
    "5min": 61,
    "15min": 21
}

class ReturnDDEvaluater(DDEvaluater):
    """ReturnDDEvaluater class."""
    def __init__(
        self,
        seed: int = 42,
        resample_rule: str = "1min",
        ticker_path_dic: dict[str | int, Path] = {},
    ) -> None:
        """initialization."""
        super().__init__(seed, ticker_path_dic)
        self.resample_rule: str = resample_rule

    def _read_csvs(self, dfs_path: Path, choose_full_size_df: bool) -> list[DataFrame]:
        """Read CSV files from a path.
        
        Args:
            dfs_path (Path): The path of the target CSV files.
            
        Returns:
            dfs (list[DataFrame]): The list of the loaded DataFrames.
        """
        dfs: list[DataFrame] = []
        for csv_path in dfs_path.glob("*.csv"):
            df: DataFrame = pd.read_csv(csv_path, index_col=0)
            if choose_full_size_df:
                if len(df) == freq_ohlcv_size_dic[self.resample_rule]:
                    dfs.append(df)
            else:
                dfs.append(df)
        return dfs
    
    def _calc_return_arr_from_df(
        self,
        ohlcv_df: DataFrame,
        colname: str = "close",
        norm: bool = True
    ) -> ndarray:
        """calculate return array from a DataFrame.

        OHLCV dataframe has price series in "OHLC" columns.
        This method picks up one of them as ohlcv_df[colname] and get price_arr [p_1,...,p_T].
        return_arr is calculated as [log(p_2/p_1),...,log(p_T/p_{T-1})].

        Args:
            ohlcv_df (DataFrame): The DataFrame of the OHLCV data.
            colname (str): The name of the column.
            norm (bool): Whether to normalize the return array.

        Returns:
            return_arr (ndarray): The return array.
        """
        price_arr: ndarray = ohlcv_df[colname].dropna().values.flatten()
        assert np.sum((price_arr <= 0)) == 0
        return_arr: ndarray = np.log(
            price_arr[1:] / price_arr[:-1] + 1e-10
        )
        if norm:
            return_arr: ndarray = (
                return_arr - np.mean(return_arr)
            ) / (np.std(return_arr) + 1e-10)
        return return_arr
    
    def _calc_return_arr_from_dfs(
        self,
        ohlcv_dfs: list[DataFrame],
        colname: str,
        norm: bool = True
    ) -> ndarray:
        """calculate return array from DataFrames."""
        return_arrs: list[ndarray] = [
            self._calc_return_arr_from_df(df, colname, norm) for df in ohlcv_dfs
        ]
        return_arr: ndarray = np.concatenate(return_arrs)
        return return_arr

    def get_point_cloud_from_path(
        self,
        num_points: int,
        ohlcv_dfs_path: Path,
        choose_full_size_df: bool = True
    ) -> ndarray:
        """Get a point cloud from a path.

        A point is defined as a logarithmic return of the stock prices.
        
        Args:
            num_points (int): The number of points in the point cloud.
            ohlcv_dfs_path (Path): The path of the target OHLCV dfs.
            
        Returns:
            point_cloud (ndarray): The point cloud.
        """
        ohlcv_dfs: list[DataFrame] = self._read_csvs(ohlcv_dfs_path, choose_full_size_df)
        return_arr: ndarray = self._calc_return_arr_from_dfs(ohlcv_dfs, "close")
        point_cloud: ndarray = self.prng.choice(
            return_arr, num_points, replace=False
        )
        point_cloud: ndarray = point_cloud.reshape(-1, 1)
        return point_cloud
    
    def draw_points(
        self,
        tickers: list[str | int],
        num_points: int,
        xlim: list[float, float],
        xlabel: str,
        save_path: Path,
        is_all_in_one_subplot: bool = True,
        subplots_arrangement: tuple[int, int] = (1, 1),
    ) -> None:
        """Draw all point clouds in 1 figure.
        
        Args:
            tickers (list[str | int]): The list of the tickers.
            num_points (int): The number of points in each point cloud.
            save_path (Path): The path to save the figure.
            is_all_in_one_subplot (bool): Whether not to devide subplots for different point clouds.
            subplots_arrangement (tuple[int, int]): The arrangement of subplots. (nrows, ncols)
        """
        nrows: int = subplots_arrangement[0]
        ncols: int = subplots_arrangement[1]
        fig: Figure = plt.figure(figsize=(10*ncols, 5*nrows), dpi=50)
        if is_all_in_one_subplot:
            ax: Optional[Axes] = fig.add_subplot(111)
        else:
            ax: Optional[Axes] = None
            if nrows * ncols < len(tickers):
                raise ValueError("The number of subplots is less than the number of tickers.")
        for i, ticker in enumerate(tickers):
            point_cloud: ndarray = self.get_point_cloud_from_ticker(
                ticker, num_points, save2dic=False
            )
            if not is_all_in_one_subplot:
                ax = fig.add_subplot(nrows, ncols, i+1)
            ax.set_xlim(xlim)
            self._draw_points(
                ax, point_cloud, draw_dims=None, label=ticker
            )
            if (
                is_all_in_one_subplot or
                i % ncols == 0
            ):
                ax.set_ylabel("Frequency")
            if (
                is_all_in_one_subplot or
                ncols * (nrows-1) <= i
            ):
                ax.set_xlabel(xlabel)
            ax.legend() if not is_all_in_one_subplot else None
        ax.legend() if is_all_in_one_subplot else None
        parent_path: Path = save_path.parent
        if not parent_path.exists():
            parent_path.mkdir(parents=True)
        plt.savefig(save_path)
        plt.close()

class TailReturnDDEvaluater(ReturnDDEvaluater):
    """TailReturnDDEvaluater class."""
    def _get_tail_return(
        self,
        sorted_return_arr: ndarray,
        cut_off_th: float = 0.05
    ) -> ndarray:
        """Calculate tail return ratios."""
        assert len(sorted_return_arr.shape) == 1
        if np.sum(sorted_return_arr != np.sort(sorted_return_arr)) != 0:
            raise ValueError(
                "sorted_return_arr must be ascendinglly sorted"
            )
        cut_sorted_return_arr: ndarray = sorted_return_arr[
            int(np.floor(len(sorted_return_arr) * (1-cut_off_th))):
        ]
        return cut_sorted_return_arr

    def get_point_cloud_from_path(
        self,
        num_points: int,
        ohlcv_dfs_path: Path,
        choose_full_size_df: bool = True,
        cut_off_th: float = 0.05
    ) -> ndarray:
        """Get a point cloud from a path.

        A point is defined as a logarithmic return of the stock prices.
        
        Args:
            num_points (int): The number of points in the point cloud.
            ohlcv_dfs_path (Path): The path of the target OHLCV dfs.
            
        Returns:
            point_cloud (ndarray): The point cloud.
        """
        ohlcv_dfs: list[DataFrame] = self._read_csvs(ohlcv_dfs_path, choose_full_size_df)
        return_arr: ndarray = self._calc_return_arr_from_dfs(ohlcv_dfs, "close")
        if len(return_arr) < num_points / cut_off_th:
            raise ValueError(
                "The number of points in the point cloud is less than num_points/cut_off_th."
            )
        return_arr: ndarray = self.prng.choice(
            return_arr, int(num_points/cut_off_th), replace=False
        )
        abs_return_arr: ndarray = np.abs(return_arr)
        sorted_return_arr: ndarray = np.sort(abs_return_arr)
        point_cloud: ndarray = self._get_tail_return(sorted_return_arr)
        point_cloud: ndarray = point_cloud.reshape(-1, 1)
        return point_cloud
    
class RVsDDEvaluater(DDEvaluater):
    """RVsDDEvaluater class."""
    def get_point_cloud_from_path(
        self,
        num_points: int,
        ohlcv_df_path: Path,
        colname: str = "close",
        resample_rule: str = "1min",
    ) -> tuple[ndarray]:
        """Get a point cloud from a path.

        A point is defined as following 4 dimensional vector:
            [log RV_t, r_t, log RV_{t+1}, r_t]
        r_t and RV_t are calculated as follows.
            1. Get price series price_arr from ohlcv_df_path 
                whose length is num_days * num_daily_obs.
            2. Reshape price_arr into num_days x num_daily_obs matrix.
                (i, j) component of price_arr p_{i,j} is a price at i-th day and j-th observation.
            3. Calculate intraday returns r_{i,j} = log(p_{i,j+1}/p_{i,j}) and create
                return matrix return_arr whose shape is (num_days, num_daily_obs-1).
            4. Calculate daily return and daily realized volatility as follows.
                r_t = r_{t,1} + ... + r_{t,num_daily_obs-1},
                RV_t = r_{t,1}^2 + ... + r_{t,num_daily_obs-1}^2.

        Args:
            num_points (int): The number of points in the point cloud.
            ohlcv_df_path (Path): The path of the target OHLCV df.
            colname (str): The name of the target column.
            resample_rule (str): The resample rule.

        Returns:
            point_cloud (ndarray): The point cloud.
        """
        num_daily_obs: int = freq_ohlcv_size_dic[resample_rule]
        ohlcv_df: DataFrame = pd.read_csv(ohlcv_df_path, index_col=0)
        price_arr: ndarray = ohlcv_df[colname].values.flatten()
        num_days: int = len(price_arr) // num_daily_obs
        if num_days * num_daily_obs != len(price_arr):
            raise ValueError("Length of price series is not multiplicatable by num_daily_obs.")
        price_arr: ndarray = price_arr.reshape(num_days, num_daily_obs)
        intraday_return_arr: ndarray = (
            np.log(price_arr[:, 1:]) - np.log(price_arr[:, :-1])
        ) * 100
        daily_return_arr: ndarray = np.sum(intraday_return_arr, axis=1).flatten()
        assert num_days == len(daily_return_arr)
        daily_rv_arr: ndarray = np.sum(intraday_return_arr**2, axis=1).flatten()
        daily_log_rv_arr: ndarray = np.log(daily_rv_arr + 1e-10)
        assert num_days == len(daily_log_rv_arr)
        point_cloud: ndarray = np.concatenate(
            [
                daily_log_rv_arr[:-1].reshape(-1, 1),
                daily_return_arr[:-1].reshape(-1, 1),
                daily_log_rv_arr[1:].reshape(-1, 1),
                daily_return_arr[1:].reshape(-1, 1),
            ],
            axis=1
        )
        assert point_cloud.shape == (num_days-1, 4)
        indices: ndarray = self.prng.choice(
            np.arange(num_days-1), num_points, replace=False
        )
        point_cloud: ndarray = point_cloud[indices, :]
        return point_cloud
