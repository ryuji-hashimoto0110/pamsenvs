from .data_distance_evaluater import DDEvaluater
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path

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
    ) -> tuple[ndarray]:
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
    ) -> tuple[ndarray]:
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
        