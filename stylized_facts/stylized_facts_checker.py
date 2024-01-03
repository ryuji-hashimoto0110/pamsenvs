import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from scipy.stats import kurtosis, kurtosistest
from typing import Optional
import warnings

class StylizedFactsChecker:
    def __init__(
        self,
        olhcv_dfs_path: Optional[Path] = None,
        orderbook_dfs_path: Optional[Path] = None,
        tick_num: Optional[int] = None
    ) -> None:
        """initialization.

        load dataframes.

        Args:
            olhcv_dfs_path (Optional[Path]): path in which OLHCV csv datas are saved. Defaults to None.
                OLHCV data consists of 5 columns: open, low, high, close, volume.
            orderbook_dfs_path (Optional[Path]): path in which limit order book csv datas are saved. Defaults to None.
                order book data consists of 2*(tick_num+1) columns: sell+tick_num, ..., +0, buy-0,...,-tick_num.
        """
        self.olhcv_dfs: list[DataFrame] = []
        if olhcv_dfs_path is not None:
            self.olhcv_dfs = self._read_csvs(olhcv_dfs_path, index_col=0)
            for df in self.olhcv_dfs:
                assert len(df.columns) == 5
                df.columns = ["open", "low", "high", "close", "volume"]
        self.orderbook_dfs: list[DataFrame] = []
        if orderbook_dfs_path is not None:
            self.orderbook_dfs = self._read_csvs(orderbook_dfs_path, index_col=0)
            if tick_num is None:
                raise ValueError(
                    "Specify tick_num."
                )
            for df in self.orderbook_dfs:
                assert len(df.columns) == 2 * (tick_num + 1)
                df.columns = (
                    [f"sell+{t}" for t in reversed(range(tick_num+1))]
                ) + (
                    [f"buy-{t}" for t in range(tick_num+1)]
                )
        self.return_arr: Optional[ndarray] = None

    def _read_csvs(
        self,
        csvs_path: Path,
        index_col: Optional[int] = None
    ) -> list[DataFrame]:
        """read all csv files in given folder path.
        """
        dfs: list[DataFrame] = []
        for csv_path in sorted(csvs_path.iterdir()):
            if csv_path.suffix == "csv":
                if index_col is not None:
                    dfs.append(pd.read_csv(csv_path, index_col=index_col))
                else:
                    dfs.append(pd.read_csv(csv_path))
        return dfs

    def _is_stacking_possible(
        self,
        dfs: list[DataFrame],
        colname: str
    ) -> bool:
        """check if it is possible to stack given column in dfs into ndarray.

        Return True if all follwing conditions hold true.
            - column named colname exists in all dataframes.
            - length of all dataframes are the same.
            - number of NaN in df[colname] of all dataframes are the same.

        Args:
            dfs (list[DataFrame]): list whose elements are dataframe. Ex: self.olhcv_dfs
            colname (str): column name to check if stacking is possible.
        """
        for df in dfs:
            if colname not in df.columns:
                return False
        if [len(df) for df in dfs].count(len(dfs[0])) != len(dfs):
            return False
        if [df[colname].isnull().sum() for df in dfs].count(dfs[0][colname].isnull().sum()) != len(dfs):
            return False
        return True

    def _stack_dfs(
        self,
        dfs: list[DataFrame],
        colname: str
    ) -> ndarray:
        """stack specified column of all dataframes.

        Args:
            dfs (list[DataFrame]): list whose elements are dataframe. Ex: self.olhcv_dfs
            colname (str): column name to stack.

        Returns:
            stacked_arr (ndarray): array whose shape is (len(dfs), len(dfs[0]))
        """
        assert self._is_stacking_possible(dfs, colname)
        col_arrs: list[ndarray] = [df[colname].dropna().values for df in dfs]
        stacked_arr: ndarray = np.stack(col_arrs, axis=0)
        assert stacked_arr.shape == (len(col_arrs), len(col_arrs[0]))
        return stacked_arr

    def check_kurtosis(self) -> tuple[ndarray, ndarray]:
        """check the kurtosis of given price series.

        Kurtosis of stock returns is generally said to be greater than 3
        (0 if calculated according to the terms with fisher).
        This method calculate kurtosises of each time series data and test if
        each kurtosis is greater than that of gaussian distribution.

        References:
            - Mandelbrot, B. (1967). The Variation of Certain Speculative Prices,
            Journal of Business, 36 (3), 394-419. http://www.jstor.org/stable/2350970
            - Fama, E. (1965). The Behaviour of Stock Market Prices. Journal of Business,
            Journal of Business, 38 (1), 34-105. https://www.jstor.org/stable/2350752

        Returns:
            kurtosis_arr (ndarray): list of kurtosises.
            pvalues_arr (ndarray): list of p-values.
        """
        if self._is_stacking_possible(self.olhcv_dfs):
            if self.returns_arr is not None:
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.returns_arr)
            else:
                prices_arr: ndarray = self._stack_dfs(self.olhcv_dfs, "close")
                self.returns_arr: ndarray = np.log(
                    prices_arr[:,1:] / prices_arr[:,:-1] + 1e-10
                )
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.returns_arr)
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            kurtosises: list[float] = []
            pvalues: list[float] = []
            for olhcv_df in self.olhcv_dfs:
                prices_arr: ndarray = olhcv_df["close"].dropna().values
                returns_arr: ndarray = np.log(
                    prices_arr[1:] / prices_arr[:-1] + 1e-10
                )[np.newaxis,:]
                kurtosis, pvalue = self._calc_kurtosis(returns_arr)
                kurtosises.append(kurtosis.item())
                pvalues.append(pvalue.item())
            kurtosis_arr: ndarray = np.array(kurtosis)[np.newaxis,:]
            pvalues_arr: ndarray = np.array(pvalue)[np.newaxis,:]
        return kurtosis_arr, pvalues_arr

    def _calc_kurtosis(
        self,
        returns_arr: ndarray,
        is_fisher: bool = True
    ) -> tuple[ndarray, ndarray]:
        """calculate kurtosis of each time series array.

        Args:
            returns_arr (ndarray): return array whose shape is
                (number of data, length of time series).
            is_fisher (bool): Defaults to True.

        Returns:
            ndarray: _description_
        """
        if len(returns_arr.shape) != 2:
            raise ValueError(
                "The shape of returns_arr must be (number of data, length of time series)."
            )
        kurtosis_arr: ndarray = kurtosis(
            returns_arr, axis=1, fisher=is_fisher, keepdims=True
        )
        pvalue_arr: ndarray = kurtosistest(
            returns_arr, axis=1, alternative="greater"
        )[1][np.newaxis,:]
        return kurtosis_arr, pvalue_arr

    








