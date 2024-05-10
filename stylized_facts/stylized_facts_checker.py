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
        ohlcv_dfs_path: Optional[Path] = None,
        specific_name: Optional[str] = None,
        need_resample: bool = False,
        figs_save_path: Optional[Path] = None
    ) -> None:
        """initialization.

        load dataframes.

        Args:
            ohlcv_dfs_path (Optional[Path]): path in which ohlcv csv datas are saved. Defaults to None.
                ohlcv data consists of 5 columns: open, high, low, close, volume.
            specific_name (Optional[str]): the specific name in csv file name. Files that contain specific_name
                are collected if this argument is specified by _read_csvs.
            need_resample (bool): whether resampling is needed. need_resample must be True when the target data
                is tick data.
            figs_save_path (Optional[Path]): path to save figures.
        """
        self.ohlcv_dfs: list[DataFrame] = []
        self.specific_name: Optional[str] = specific_name
        if ohlcv_dfs_path is not None:
            self.ohlcv_dfs = self._read_csvs(
                ohlcv_dfs_path,
                need_resample=need_resample,
                index_col=0
            )
            for df in self.ohlcv_dfs:
                if len(df.columns) == 5:
                    df.columns = ["open", "high", "low", "close", "volume"]
                elif len(df.columns) == 6:
                    df.columns = ["open", "high", "low", "close", "volume", "num_events"]
                    df["num_events"] = df["num_events"] / df["num_events"].sum()
                df["volume"] = df["volume"] / df["volume"].sum()
        self.return_arr: Optional[ndarray] = None
        self.figs_save_path: Optional[Path] = figs_save_path

    def _read_csvs(
        self,
        csvs_path: Path,
        need_resample: bool,
        index_col: Optional[int] = None
    ) -> list[DataFrame]:
        """read all csv files in given folder path.
        """
        dfs: list[DataFrame] = []
        for csv_path in sorted(csvs_path.rglob("*.csv")):
            if self.specific_name is not None:
                csv_name: str = csv_path.name
                if self.specific_name in csv_name:
                    continue
            if index_col is not None:
                df: DataFrame = pd.read_csv(csv_path, index_col=index_col)
            else:
                df: DataFrame = pd.read_csv(csv_path)
            if need_resample:
                df = self._resample(df)
            dfs.append(df)
        return dfs

    def _resample(self, df: DataFrame) -> DataFrame:
        """resample tick data to OHLCV data.

        Args:
            df (DataFrame): dataframe of tick data. df must have at least "market_price", "num_events" columns.
        """
        assert "market_price" in df.columns
        assert "num_events" in df.columns
        df.index = pd.to_datetime(df.index)
        resampled_df: DataFrame = df["market_price"].resample(rule="min").ohlc()
        resampled_df["volume"] = df["event_volume"].resample(rule="min").apply(sum)
        resampled_df["num_events"] = df["event_volume"].resample(rule="min").count()
        return resampled_df

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
            dfs (list[DataFrame]): list whose elements are dataframe. Ex: self.ohlcv_dfs
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
            dfs (list[DataFrame]): list whose elements are dataframe. Ex: self.ohlcv_dfs
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
        ohlcv_df: DataFrame,
        colname: str
    ) -> ndarray:
        """convert price time series to return time series from 1 dataframe.
        """
        price_arr: ndarray = ohlcv_df[colname].dropna().values
        assert np.sum((price_arr <= 0)) == 0
        return_arr: ndarray = np.log(
            price_arr[1:] / price_arr[:-1] + 1e-10
        )[np.newaxis,:]
        return return_arr

    def _calc_return_arr_from_dfs(
        self,
        ohlcv_dfs: list[DataFrame],
        colname: str
    ) -> ndarray:
        """convert price time series to return time series from dataframes list.
        """
        price_arr: ndarray = self._stack_dfs(ohlcv_dfs, colname)
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
        if self._is_stacking_possible(self.ohlcv_dfs, "close"):
            if self.return_arr is not None:
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.return_arr)
            else:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.ohlcv_dfs, "close"
                )
                kurtosis_arr, pvalue_arr = self._calc_kurtosis(self.return_arr)
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            kurtosises: list[float] = []
            pvalues: list[float] = []
            for ohlcv_df in self.ohlcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(ohlcv_df, "close")
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
        in real markets (universal cubic law).
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
        if self._is_stacking_possible(self.ohlcv_dfs, "close"):
            if self.return_arr is None:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.ohlcv_dfs, "close"
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
            for ohlcv_df in self.ohlcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(ohlcv_df, "close")
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
        sorted_return_arr: ndarray = np.sort(return_arr, axis=1)
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
    ) -> None:
        """draw CCDF of return distribution by log-log scale.

        Complementary cumulative distribution function (CCDF) is defined as P[x<X], namely
        defined as the probability that stochastic variable X is greater than a certain
        threshold x.
        CCDF is used to see the tail of samples that seems to be fitted by power law. Here,
        using CCDF, one can check visually that return distribution is fat-tailed.

        Args:
            ax (Optional[Axes]): ax to draw figure. default to None.
            label: label
            color: color
            save_name (Optional[str]): file name to save figure. Default to None.
            draw_idx (Optional[int]): If draw_idx is specified, price data of
                self.ohlcv_dfs[draw_idx] is only chosen to draw figure. Otherwise, all data
                are concatted and used to draw. Defaults to None.
        """
        if draw_idx is None:
            if self._is_stacking_possible(self.ohlcv_dfs, "close"):
                if self.return_arr is None:
                    self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                        self.ohlcv_dfs, "close"
                    )
                return_arr: ndarray = self.return_arr.flatten()
            else:
                warnings.warn(
                    "Could not stack dataframe. Maybe the lengths of dataframes differ." + \
                    "Following procedure may takes time..."
                )
                return_arrs: list[ndarray] = []
                for ohlcv_df in self.ohlcv_dfs:
                    return_arrs.append(
                        self._calc_return_arr_from_df(ohlcv_df, "close").flatten()
                    )
                return_arr: ndarray = np.concatenate(return_arrs)
        else:
            return_arr: ndarray = self._calc_return_arr_from_df(
                self.ohlcv_dfs[draw_idx], "close"
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
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("return")
        ax.set_ylabel("CCDF")
        ax.set_title("Complementary Cumulative Distribution Function (CCDF) of absolute price returns")
        if save_name is not None:
            if self.figs_save_path is None:
                raise ValueError(
                    "specify directory: self.figs_save_path"
                )
            save_path: Path = self.figs_save_path / save_name
            plt.savefig(str(save_path))

    def check_autocorrelation(self, lags: list[int]) -> dict[int, ndarray]:
        """_summary_

        Args:
            lags (list[int]): _description_

        Returns:
        """
        if self._is_stacking_possible(self.ohlcv_dfs, "close"):
            if self.return_arr is None:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.ohlcv_dfs, "close"
                )
            acorr_dic: dict[int, ndarray] = self._calc_autocorrelation(
                np.abs(self.return_arr), lags
            )
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            acorr_l_dic: dict[int, list[float]] = {lag: [] for lag in lags}
            for ohlcv_df in self.ohlcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(ohlcv_df, "close")
                acorr_dic_: dict[int, float] = self._calc_autocorrelation(
                    np.abs(return_arr), lags
                )
                for lag in lags:
                    acorr_l_dic[lag].append(acorr_dic_[lag].item())
            acorr_dic: dict[int, ndarray] = {}
            for lag, acorrs in acorr_l_dic.items():
                acorr_dic[lag] = np.array(acorrs)[:,np.newaxis]
        return acorr_dic

    def _calc_autocorrelation(
        self,
        abs_return_arr: ndarray,
        lags: list[int]
    ) -> dict[int, ndarray]:
        """_summary_

        Args:
            return_arr (ndarray): _description_
            lags (list[int]): _description_

        Returns:
            dict[int, ndarray]: _description_
        """
        acorr_dic: list[int, ndarray] = {}
        for lag in lags:
            abs_mean: ndarray = np.mean(abs_return_arr, axis=1, keepdims=True)
            acov: ndarray = np.mean(
                (abs_return_arr[:,lag:]-abs_mean)*(abs_return_arr[:,:-lag]-abs_mean),
                axis=1, keepdims=True
            )
            var: ndarray = np.var(abs_return_arr, axis=1, keepdims=True)
            acorr_dic[lag] = acov / (var + 1e-10)
        return acorr_dic

    def check_volume_volatility_correlation(self) -> ndarray:
        if self._is_stacking_possible(self.ohlcv_dfs, "close"):
            volume_arr: ndarray = self._stack_dfs(
                self.ohlcv_dfs, "volume"
            )
            volume_arr = volume_arr[:,1:]
            if self.return_arr is None:
                self.return_arr: ndarray = self._calc_return_arr_from_dfs(
                    self.ohlcv_dfs, "close"
                )
            corr_arr: ndarray = self._calc_volume_volatility_correlation(
                np.abs(self.return_arr), volume_arr
            )
        else:
            warnings.warn(
                "Could not stack dataframe. Maybe the lengths of dataframes differ. Following procedure may takes time..."
            )
            corrs: list[float] = []
            for ohlcv_df in self.ohlcv_dfs:
                return_arr: ndarray = self._calc_return_arr_from_df(ohlcv_df, "close")
                volume_arr: ndarray = ohlcv_df["volume"].values[np.newaxis,:]
                corrs.append(
                    self._calc_volume_volatility_correlation(
                        np.abs(return_arr), volume_arr
                    ).item()
                )
            corr_arr: ndarray = np.array(corrs)[np.newaxis,:]
        return corr_arr

    def _calc_volume_volatility_correlation(
        self,
        abs_return_arr: ndarray,
        volume_arr: ndarray
    ) -> ndarray:
        """_summary_

        Args:
            abs_return_arr (ndarray): _description_ (number of data, length of time series).
            volume_arr (ndarray): _description_ (number of data, length of time series).

        Returns:
            ndarray: _description_ (number of data, 1).
        """
        abs_return_mean: ndarray = np.mean(
            abs_return_arr, axis=1, keepdims=True
        )
        volume_mean: ndarray = np.mean(
            volume_arr, axis=1, keepdims=True
        )
        abs_retrurn_std: ndarray = np.std(
            abs_return_arr, axis=1, keepdims=True
        )
        volume_std: ndarray = np.std(
            volume_arr, axis=1, keepdims=True
        )
        volume_volatility_correlation: ndarray = np.mean(
            (abs_return_arr - abs_return_mean) * (volume_arr - volume_mean),
            axis=1, keepdims=True
        ) / (
            abs_retrurn_std * volume_std + 1e-10
        )
        return volume_volatility_correlation

    def check_stylized_facts(
        self,
        save_path: Path
    ) -> None:
        if 0 < len(self.ohlcv_dfs):
            kurtosis_arr, p_values = self.check_kurtosis()
            left_tail_arr, right_tail_arr = self.check_hill_index()
            volume_volatility_correlation = self.check_volume_volatility_correlation()
            acorr_dic: dict[int, ndarray] = self.check_autocorrelation(
                [lag for lag in range(1,31)]
            )
            data_dic: dict[str, ndarray]= {
                "kurtosis": kurtosis_arr.flatten(),
                "kurtosis_p": p_values.flatten(),
                "tail (left)": left_tail_arr.flatten(),
                "tail (right)": right_tail_arr.flatten(),
                "vv_corr": volume_volatility_correlation.flatten()
            }
            for lag, acorr in acorr_dic.items():
                data_dic[f"acorr lag{lag}"] = acorr.flatten()
            stylized_facts_df: DataFrame = pd.DataFrame(data_dic)
            stylized_facts_df.to_csv(str(save_path))
