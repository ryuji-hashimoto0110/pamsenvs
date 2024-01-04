import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
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
        tick_num: Optional[int] = None,
        figs_save_path: Optional[Path] = None
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
        self.figs_save_path: Optional[Path] = figs_save_path

    def _read_csvs(
        self,
        csvs_path: Path,
        index_col: Optional[int] = None
    ) -> list[DataFrame]:
        """read all csv files in given folder path.
        """
        dfs: list[DataFrame] = []
        for csv_path in sorted(csvs_path.iterdir()):
            if csv_path.suffix == ".csv":
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

    def _calc_return_arr_from_df(
        self,
        olhcv_df: DataFrame,
        colname: str
    ) -> ndarray:
        """convert price time series to return time series from 1 dataframe.
        """
        price_arr: ndarray = olhcv_df[colname].dropna().values
        assert np.sum((price_arr <= 0)) == 0
        return_arr: ndarray = np.log(
            price_arr[1:] / price_arr[:-1] + 1e-10
        )[np.newaxis,:]
        return return_arr

    def _calc_return_arr_from_dfs(
        self,
        olhcv_dfs: list[DataFrame],
        colname: str
    ) -> ndarray:
        """convert price time series to return time series from dataframes list.
        """
        price_arr: ndarray = self._stack_dfs(olhcv_dfs, colname)
        assert np.sum((price_arr <= 0)) == 0
        return_arr: ndarray = np.log(
            price_arr[:,1:] / price_arr[:,:-1] + 1e-10
        )
        return return_arr

    def check_kurtosis(self) -> tuple[ndarray, ndarray]:
        """check the kurtosis of given price time series.

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
            kurtosis_arr (ndarray): kurtosises. (number of data,1)
            pvalues_arr (ndarray): p-values. (number of data,1)
        """
        if self._is_stacking_possible(self.olhcv_dfs, "close"):
            if self.return_arr is not None:
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.return_arr)
            else:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.olhcv_dfs, "close"
                )
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.return_arr)
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            kurtosises: list[float] = []
            pvalues: list[float] = []
            for olhcv_df in self.olhcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(olhcv_df, "close")
                kurtosis, pvalue = self._calc_kurtosis(return_arr)
                kurtosises.append(kurtosis.item())
                pvalues.append(pvalue.item())
            kurtosis_arr: ndarray = np.array(kurtosis)[:,np.newaxis]
            pvalue_arr: ndarray = np.array(pvalue)[:,np.newaxis]
        return kurtosis_arr, pvalue_arr

    def _calc_kurtosis(
        self,
        return_arr: ndarray,
        is_fisher: bool = True
    ) -> tuple[ndarray, ndarray]:
        """calculate kurtosis of each time series array.

        Args:
            return_arr (ndarray): return array whose shape is
                (number of data, length of time series).
            is_fisher (bool): Defaults to True.

        Returns:
            kurtosis_arr (ndarray): kurtosises. (number of data, 1)
            pvalues_arr (ndarray): p-values. (number of data, 1)
        """
        if len(return_arr.shape) != 2:
            raise ValueError(
                "The shape of return_arr must be (number of data, length of time series)."
            )
        kurtosis_arr: ndarray = kurtosis(
            return_arr, axis=1, fisher=is_fisher, keepdims=True
        )
        pvalue_arr: ndarray = kurtosistest(
            return_arr, axis=1, alternative="greater"
        )[1][:,np.newaxis]
        return kurtosis_arr, pvalue_arr

    def check_hill_index(
        self,
        cut_off_th: float = 0.05
    ) -> tuple[ndarray, ndarray]:
        """check Hill-tail index of given price time series.

        The stock return distribution is generally said to be fat-tail.
        According to some empirical researches, the tail index is normally around or below 3
        in real markets.
        Also, the skewness of the returns is negative. In other words,
        tail due to negative returns is fatter than that due to positive returns.

        Note: Hill Index assumes non-negative values in tail area. Therefore, to calculate
        both left and right tail indices, the mean of return distribution must be in near 0.

        References:
            - Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution,
            Annals of Statistics 3 (5), 1163-1173. https://doi.org/10.1214/aos/1176343247
            - Lux, T. (2001). The limiting extremal behaviour of speculative returns:
            an analysis of intra-daily data from the Frankfurt Stock Exchange,
            Applied Financial Economics, 11, 299-315. https://doi.org/10.1080/096031001300138708
            - Gabaix, X., Gopikrishnan, P., Plerou, V., Stanley, H. E. (2003).
            A theory of power-low distributions in financial market fluctuations,
            Nature 423, 267-270. http://dx.doi.org/10.1038/nature01624

        Args:
            cut_off_th (float): threshold to cut-off samples inside tail of the distributions.
                Default to 0.05.

        Returns:
            left_tail_arr (ndarray): tail indices of left side of samples.
            right_tail_arr (ndarray): tail indices of right side of samples.
        """
        assert 0 < cut_off_th and cut_off_th < 1
        if self._is_stacking_possible(self.olhcv_dfs, "close"):
            if self.return_arr is None:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.olhcv_dfs, "close"
                )
            left_tail_arr, right_tail_arr = self._calc_both_sides_hill_indices(
                self.return_arr, cut_off_th
            )
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            left_tails: list[float] = []
            right_tails: list[float] = []
            for olhcv_df in self.olhcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(olhcv_df, "close")
                left_tail_arr, right_tail_arr = self._calc_both_sides_hill_indices(
                    return_arr, cut_off_th
                )
                left_tails.append(left_tail_arr.item())
                right_tails.append(right_tail_arr.item())
            left_tail_arr: ndarray = np.array(left_tails)[np.newaxis,:]
            right_tail_arr: ndarray = np.array(right_tails)[np.newaxis,:]
        return left_tail_arr, right_tail_arr

    def _calc_hill_indices(
        self,
        sorted_return_arr: ndarray,
        cut_off_th: float = 0.05
    ) -> ndarray:
        """calculate right side Hill tail indices of ascendinglly sorted return array.

        Args:
            sorted_return_arr (ndarray): return array whose shape is
                (number of data, length of time series).
                This array must be ascendinglly sorted.
            cut_off_th (float): threshold to cut-off samples inside tail of the distributions.
                Default to 0.05.

        Returns:
            tail_arr (ndarray): tail indices. (number of data, 1)
        """
        assert len(sorted_return_arr.shape) == 2
        if np.sum(sorted_return_arr != np.sort(sorted_return_arr, axis=1)) != 0:
            raise ValueError(
                "sorted_return_arr must be ascendinglly sorted"
            )
        sorted_return_arr: ndarray = sorted_return_arr[
            :,int(np.floor(sorted_return_arr.shape[1] * (1-cut_off_th))):
        ]
        if np.sum(sorted_return_arr <= 0) != 0:
            raise ValueError(
                "Non positive elements found in tail area of sorted_return_arr. Maybe you should reduce cut_off_th."
            )
        k: int = sorted_return_arr.shape[1]
        tail_arr: ndarray = 1 / k * np.sum(
            np.log(sorted_return_arr[:,1:] / sorted_return_arr[:,0][:,np.newaxis]),
            axis=1
        )[:,np.newaxis]
        return tail_arr

    def _calc_both_sides_hill_indices(
        self,
        return_arr: ndarray,
        cut_off_th: float = 0.05
    ) -> tuple[ndarray, ndarray]:
        """_summary_

        Args:
            return_arr (ndarray): _description_
            cut_off_th (float, optional): _description_. Defaults to 0.05.

        Returns:
            left_tail_arr (ndarray): _description_
            right_tail_arr (ndarray): _description_
        """
        sorted_return_arr: ndarray = np.sort(self.return_arr, axis=1)
        right_tail_arr: ndarray = self._calc_hill_indices(
            sorted_return_arr, cut_off_th
        )
        minus_return_arr: ndarray = - 1 * return_arr
        sorted_minus_return_arr: ndarray = np.sort(minus_return_arr, axis=1)
        left_tail_arr: ndarray = self._calc_hill_indices(
            sorted_minus_return_arr, cut_off_th
        )
        return left_tail_arr, right_tail_arr

    def check_ccdf(
        self,
        ax: Optional[Axes] = None,
        label: str = "CCDF",
        color: str = "black",
        save_name: Optional[str] = None,
        draw_idx: Optional[int] = None
    ) -> tuple[ndarray, ndarray]:
        """draw CCDF of return distribution by log-log scale.

        Complementary, cumulative distribution function (CCDF) is defined as P[x<X], namely
        defined as the probability that stochastic variable X is greater than a certain
        threshold x.
        CCDF is used to see the tail of samples that seems to be fitted by power law. Here,
        we can check visually that return distribution is fat-tailed using CCDF.

        Args:
            ax (Optional[Axes]): ax to draw figure. default to None.
            label: label
            color: color
            save_name (Optional[str]): file name to save figure. Default to None.
            draw_idx (Optional[int]): If draw_idx is specified, price data of
                self.olhcv_dfs[draw_idx] is only chosen to draw figure. Otherwise, all data
                are concatted and used to draw. Defaults to None.
        """
        if draw_idx is None:
            if self._is_stacking_possible(self.olhcv_dfs, "close"):
                if self.return_arr is None:
                    self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                        self.olhcv_dfs, "close"
                    )
                return_arr: ndarray = self.return_arr.flatten()
            else:
                warnings.warn(
                    "Could not stack dataframe. Maybe the lengths of dataframes differ." + \
                    "Following procedure may takes time..."
                )
                return_arrs: list[ndarray] = []
                for olhcv_df in self.olhcv_dfs:
                    return_arrs.append(
                        self._calc_return_arr_from_df(olhcv_df, "close").flatten()
                    )
                return_arr: ndarray = np.concatenate(return_arrs)
        else:
            return_arr: ndarray = self._calc_return_arr_from_df(
                self.olhcv_dfs[draw_idx], "close"
            ).flatten()
        assert len(return_arr.shape) == 1
        sorted_abs_return_arr: ndarray = np.sort(np.abs(return_arr))
        ccdf: ndarray = 1 - (
            1 + np.arange(len(sorted_abs_return_arr))
        ) / len(sorted_abs_return_arr)
        if ax is None:
            fig = plt.figure(figsize=(10,6))
            ax: Axes = fig.add_subplot(1,1,1)
        ax.plot(sorted_abs_return_arr, ccdf, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("return")
        ax.set_ylabel("CCDF")
        ax.set_title("Complementary Cumulative Distribution Function (CCDF) of price returns")
        if save_name is not None:
            if self.figs_save_path is None:
                raise ValueError(
                    "specify directory: self.figs_save_path"
                )
            save_path: Path = self.figs_save_path / save_name
            plt.savefig(str(save_path))
        return sorted_abs_return_arr, ccdf
