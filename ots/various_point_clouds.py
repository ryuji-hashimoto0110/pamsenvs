from .data_distance_evaluater import DDEvaluater
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from scipy.stats import linregress
from stylized_facts import bybit_freq_ohlcv_size_dic
from stylized_facts import flex_freq_ohlcv_size_dic
from scipy import stats
from typing import Optional
plt.rcParams["font.size"] = 20

class ReturnDDEvaluater(DDEvaluater):
    """ReturnDDEvaluater class."""
    def __init__(
        self,
        seed: int = 42,
        is_bybit: bool = False,
        resample_rule: str = "1min",
        ticker_path_dic: dict[str | int, Path] = {},
    ) -> None:
        """initialization."""
        super().__init__(seed, ticker_path_dic)
        self.is_bybit: bool = is_bybit
        self.freq_ohlcv_size_dic: dict[str, int] = \
            bybit_freq_ohlcv_size_dic if is_bybit else flex_freq_ohlcv_size_dic
        self.resample_rule: str = resample_rule
    
    def calc_statistics(
        self,
        point_cloud: ndarray,
    ) -> list[float]:
        """Calculate kurtosis of the point cloud.
        
        The point cloud for ReturnDDEvaluater is defined as a 1D array of return values.

        Args:
            point_cloud (ndarray): The point cloud.

        Returns:
            statistics (list[float]): The statistics.
        """
        kurtosis: float = stats.kurtosis(point_cloud.flatten(), fisher=True)
        statistics: list[float] = [kurtosis]
        return statistics

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
                if len(df) == self.freq_ohlcv_size_dic[self.resample_rule]:
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
        ylim: list[float, float],
        xlabel: str,
        ylabel: str,
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
            point_cloud, statistics = self.get_point_cloud_from_ticker(
                ticker, num_points, save2dic=False, return_statistics=True
            )
            if not is_all_in_one_subplot:
                ax = fig.add_subplot(nrows, ncols, i+1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            self._draw_points(
                ax, point_cloud, draw_dims=None, label=ticker + f" stat={statistics[0]:.2f}"
            )
            if (
                is_all_in_one_subplot or
                i % ncols == 0
            ):
                ax.set_ylabel(ylabel)
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
    def get_statistics(self) -> list[str]:
        return ["Tail"]
    
    def calc_statistics(self, point_cloud: ndarray) -> list[float]:
        """Calculate the tail return ratio of the point cloud.
        
        The point cloud for TailReturnDDEvaluater is defined as a 1D array of tail-return values.

        Args:
            point_cloud (ndarray): The point cloud.

        Returns:
            statistics (list[float]): The statistics.
        """
        k: int = len(point_cloud)
        tail_index: float = 1 / (1 / k * np.sum(point_cloud))
        statistics: list[float] = [tail_index]
        return statistics
        
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
        cut_sorted_return_log_ratio_arr: ndarray = np.log(
            cut_sorted_return_arr / cut_sorted_return_arr[0]
        )
        return cut_sorted_return_log_ratio_arr

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
    
class ReturnTSDDEvaluater(ReturnDDEvaluater):
    """ReturnTSDDEvaluater class."""
    def __init__(
        self,
        lags: list[int],
        seed: int = 42,
        is_bybit: bool = False,
        resample_rule: str = "1min",
        ticker_path_dic: dict[str | int, Path] = {},
    ) -> None:
        """initialization."""
        super().__init__(seed, is_bybit, resample_rule, ticker_path_dic)
        self.lags: list[int] = lags
        self.lags.sort()

    def _calc_autocorrelation(
        self,
        abs_return_arr: ndarray,
        lags: list[int],
        keepdim: bool = True,
    ) -> dict[int, ndarray | float]:
        """_summary_

        Args:
            return_arr (ndarray): _description_
            lags (list[int]): _description_

        Returns:
            dict[int, ndarray]: _description_
        """
        acorr_dic: list[int, ndarray] = {}
        for lag in lags:
            abs_mean: float | ndarray = np.mean(
                abs_return_arr, axis=1, keepdims=True
            ) if keepdim else np.mean(abs_return_arr)
            acov: ndarray | float = np.mean(
                (abs_return_arr[:,lag:]-abs_mean)*(abs_return_arr[:,:-lag]-abs_mean),
                axis=1, keepdims=True
            ) if keepdim else np.mean(
                (abs_return_arr[:,lag:]-abs_mean)*(abs_return_arr[:,:-lag]-abs_mean)
            )
            var: ndarray | float = np.var(
                abs_return_arr, axis=1, keepdims=True
            )  if keepdim else np.var(abs_return_arr)
            acorr_dic[lag] = acov / (var + 1e-10)
        return acorr_dic
    
    def get_statistics(self) -> list[str]:
        if len(self.lags) == 1:
            return [f"acorr({self.lags[0]})"]
        else:
            return ["tail (acorr)", "first negative lag"]

    def calc_statistics(self, point_cloud: ndarray) -> list[float]:
        statistics: list[float] = []
        if len(self.lags) == 1:
            acorr_dic: dict[int, ndarray] = self._calc_autocorrelation(
                np.abs(self.return_arr), self.lags, keepdim=False
            )
            statistics.append(float(acorr_dic[self.lags[0]]))
        else:
            lags: list[int] = [lag for lag in range(1,121)]
            acorr_dic: dict[int, ndarray] = self._calc_autocorrelation(
                np.abs(self.return_arr), lags, keepdim=False
            )
            first_negative_lag: int = -1
            log_lags: list[float] = []
            log_acorrs: list[float] = []
            for lag in lags:
                acorr_mean: float = np.mean(acorr_dic[lag])
                if acorr_mean < 0:
                    first_negative_lag = lag
                    break
                else:
                    log_lags.append(np.log(lag))
                    log_acorrs.append(np.log(acorr_mean))
            if first_negative_lag == 0:
                statistics.extend([-1, -1])
            else:
                log_lag_arr: ndarray = np.array(log_lags)
                log_acorr_arr: ndarray = np.array(log_acorrs)
                lr = linregress(log_lag_arr, log_acorr_arr)
                tail: float = - lr.slope
                statistics.extend([tail, first_negative_lag])
        return statistics

    def get_point_cloud_from_path(
        self,
        num_points: int,
        ohlcv_dfs_path: Path,
        choose_full_size_df: bool = True
    ) -> ndarray:
        ohlcv_dfs: list[DataFrame] = self._read_csvs(
            ohlcv_dfs_path, choose_full_size_df
        )
        return_arrs: list[ndarray] = [
            self._calc_return_arr_from_df(df, "close", norm=True) for df in ohlcv_dfs
        ]
        self.return_arr: ndarray = np.stack(return_arrs, axis=0)
        assert self.return_arr.shape == (len(ohlcv_dfs), len(return_arrs[0]))
        abs_return_arr: ndarray = np.abs(self.return_arr)
        abs_return_arrs: list[ndarray] = []
        abs_return_arrs.append(
            abs_return_arr[:,:-self.lags[-1]].flatten()
        )
        for lag in self.lags[:-1]:
            abs_return_arrs.append(
                abs_return_arr[:,lag:-self.lags[-1]+lag].flatten()
            )
        abs_return_arrs.append(
            abs_return_arr[:,self.lags[-1]:].flatten()
        )
        abs_return_arr = np.stack(abs_return_arrs, axis=1)
        indices: ndarray = self.prng.choice(
            np.arange(len(abs_return_arr)), num_points, replace=False
        )
        point_cloud: ndarray = abs_return_arr[indices]
        return point_cloud

class RVsDDEvaluater(DDEvaluater):
    """RVsDDEvaluater class."""
    def __init__(
        self,
        seed: int = 42,
        is_bybit: bool = False,
        resample_rule: str = "1min",
        ticker_path_dic: dict[str | int, Path] = {},
    ) -> None:
        """initialization."""
        super().__init__(seed, ticker_path_dic)
        self.is_bybit: bool = is_bybit
        self.freq_ohlcv_size_dic: dict[str, int] = \
            bybit_freq_ohlcv_size_dic if is_bybit else flex_freq_ohlcv_size_dic
        self.resample_rule: str = resample_rule

    def get_point_cloud_from_path(
        self,
        num_points: int,
        ohlcv_df_path: Path,
        colname: str = "close",
    ) -> tuple[ndarray]:
        """Get a point cloud from a path.

        A point is defined as following 4 dimensional vector:
            [RV_t, r_t, RV_{t+1}, r_t]
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
        num_daily_obs: int = self.freq_ohlcv_size_dic[self.resample_rule]
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
        daily_return_arr = (
            daily_return_arr - np.mean(daily_return_arr)
        ) / (np.std(daily_return_arr) + 1e-10)
        #daily_return_arr = np.clip(daily_return_arr, -7, 7)
        assert num_days == len(daily_return_arr)
        daily_rv_arr: ndarray = np.sum(intraday_return_arr**2, axis=1).flatten()
        daily_log_rv_arr: ndarray = np.log(daily_rv_arr)
        daily_log_rv_arr = (
            daily_log_rv_arr - np.mean(daily_log_rv_arr)
        ) / (np.std(daily_log_rv_arr) + 1e-10)
        assert num_days == len(daily_log_rv_arr)
        point_cloud: ndarray = np.concatenate(
            [
                daily_log_rv_arr[:-1].reshape(-1, 1),
                daily_return_arr[:-1].reshape(-1, 1),
                daily_log_rv_arr[1:].reshape(-1, 1),
            ],
            axis=1
        )
        assert point_cloud.shape == (num_days-1, 3)
        indices: ndarray = self.prng.choice(
            np.arange(num_days-1), num_points, replace=False
        )
        point_cloud: ndarray = point_cloud[indices]
        return point_cloud
