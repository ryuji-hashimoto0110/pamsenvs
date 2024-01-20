import bisect
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import pathlib
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional
import warnings
from scipy import stats

class RVDataset(Dataset):
    def __init__(
        self,
        olhcv_path: Path,
        csv_names: Optional[list[str]],
        obs_num: int,
        input_time_length: int,
        imgs_path: Optional[Path] = None
    ) -> None:
        """initialization.

        Create Dataset for realized volatility forecasting.
        initialization method create dataset by following procedure.
            1. read all csv files from olhcv_path folder.
            2. fold the time series to calculate realized volatility.
                Assume that length of a csv being time_series_length,
                then folded array is (time_series_length/obs_num, obs_num).
                Note that time_series_length should be divisible by obs_num.
            3. calculate return, volume, realized volatility.
                shape of return_arrs, volume_arrs and rv_arrs is
                (data_num, time_series_length/obs_num)
                Each of the features are normalized.

        Args:
            olhcv_path (Path): folder path where the OLHCV files to be trained are stored
            obs_num (int): number of data used to calculate a single return, volume,
                and realized volatility.
                If the time interval for high-frequency datas in olhcv_path is 5 minutes
                and obs_num is set to 12, the time interval in RVDataset is 60 minutes.
        """
        olhcv_dfs = self._read_csvs(olhcv_path, csv_names)
        self.obs_num: int = obs_num
        self.input_time_length: int = input_time_length
        price_arrs: ndarray | list[ndarray] = self._fold_dfs(olhcv_dfs, "close")
        volume_arrs: ndarray | list[ndarray] = self._fold_dfs(olhcv_dfs, "volume")
        self.return_arrs: ndarray | list[ndarray] = self._calc_return(price_arrs)
        if isinstance(self.return_arrs, ndarray):
            self.return_arrs = self._normalize(self.return_arrs)
        self.volume_arrs: ndarray | list[ndarray] = self._calc_volume(volume_arrs)
        if isinstance(self.volume_arrs, ndarray):
            self.volume_arrs = self._normalize(self.volume_arrs)
        self.rv_arrs: ndarray | list[ndarray] = self._calc_rv(price_arrs)
        if isinstance(self.rv_arrs, ndarray):
            self.rv_arrs = self._normalize(self.rv_arrs)
        self.num_data_arr: ndarray = self._count_cumsum_datas()
        self.imgs_path: Path = imgs_path

    def _read_csvs(
        self,
        csvs_path: Path,
        csv_names: Optional[list[str]]
    ) -> list[DataFrame]:
        """read all csv files in given folder path.

        Args:
            csvs_path (Path): folder path where the cav files are stored.
        """
        dfs: list[DataFrame] = []
        for csv_path in sorted(csvs_path.iterdir()):
            csv_name: str = csv_path.name
            if csv_names is not None:
                if not csv_name in csv_names:
                    continue
            if csv_path.suffix == ".csv":
                df: DataFrame = pd.read_csv(csv_path, index_col=0)
                assert len(df.columns) == 5
                df.columns = ["open", "low", "high", "close", "volume"]
                dfs.append(df)
        return dfs

    def _fold_dfs(self, dfs: list[DataFrame], colname: str) -> ndarray | list[ndarray]:
        """fold specific columns of dfs.

        Assume that N=len(dfs), m=obs_num, m*n=len(df), then
            folded_columns.shape = (N, n, m)

        Args:
            dfs (list[DataFrame]): list of OLHCV dataframes. It is recommended that
                lengths of all dataframes are the same.
            colname (str): name of the column to create features.

        Returns:
            folded_columns (ndarray | list[ndarray])
        """
        folded_columns: list[ndarray] = []
        for df in dfs:
            assert colname in df.columns
            column_arr: ndarray = df[colname].values[:,np.newaxis]
            assert len(column_arr.shape) == 2
            if len(column_arr) % self.obs_num != 0:
                new_data_num: int = int(
                    np.floor(len(column_arr) / self.obs_num) * self.obs_num
                )
                warnings.warn(
                    f"number of obs {len(column_arr)} cannot be devised by {self.obs_num}. " +
                    f"data will be pruned to {new_data_num}."
                )
                column_arr = column_arr[:new_data_num]
            folded_columns.append(column_arr.reshape(-1, self.obs_num))
        num_obses: list[int] = [folded_column.shape[0] for folded_column in folded_columns]
        if num_obses.count(num_obses[0]) == len(num_obses):
            folded_columns: ndarray = np.stack(folded_columns)
        else:
            warnings.warn(
                "failed to stack dataframes. maybe lengths of dataframes differ."
            )
        return folded_columns

    def _calc_return(self, price_arrs: ndarray | list[ndarray]) -> ndarray | list[ndarray]:
        """calculate log returns.

        Assume that N=len(dfs), m=obs_num, m*n=len(df), then return_arrs.shape = (N, n)

        Args:
            price_arrs (ndarray | list[ndarray]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            ndarray | list[ndarray]: _description_
        """
        if isinstance(price_arrs, list):
            return_arrs: list[ndarray] = []
            for price_arr in price_arrs:
                assert isinstance(price_arr, ndarray)
                log_price_arr: ndarray = np.log(price_arr)
                return_arr: ndarray = log_price_arr[:,-1] - log_price_arr[:,0]
                assert len(return_arr.shape) == 1
                return_arrs.append(return_arr)
        elif isinstance(price_arrs, ndarray):
            assert len(price_arrs.shape) == 3
            log_price_arrs: ndarray = np.log(price_arrs)
            return_arrs: ndarray = log_price_arrs[:,:,-1] - log_price_arrs[:,:,1]
        else:
            raise NotImplementedError
        return return_arrs

    def _calc_rv(self, price_arrs: ndarray | list[ndarray]) -> ndarray | list[ndarray]:
        """calculate realized volatilities.

        Assume that N=len(dfs), m=obs_num, m*n=len(df), then return_arrs.shape = (N, n)

        Args:
            price_arrs (ndarray | list[ndarray]): _description_

        Returns:
            ndarray | list[ndarray]: _description_
        """
        if isinstance(price_arrs, list):
            rv_arrs: list[ndarray] = []
            for price_arr in price_arrs:
                assert isinstance(price_arr, ndarray)
                log_price_arr: ndarray = np.log(price_arr)
                return_arr: ndarray = log_price_arr[:,1:] - log_price_arr[:,:-1]
                assert len(return_arr.shape) == 2
                rv_arrs.append(
                    np.log((return_arr**2).sum(axis=1) + 1e-10)
                )
        elif isinstance(price_arrs, ndarray):
            assert len(price_arrs.shape) == 3
            log_price_arrs: ndarray = np.log(price_arrs)
            return_arrs: ndarray = log_price_arrs[:,:,1:] - log_price_arrs[:,:,:-1]
            rv_arrs: ndarray = np.log((return_arrs**2).sum(axis=2) + 1e-10)
        else:
            raise NotImplementedError
        return rv_arrs

    def _calc_volume(self, volume_arrs: ndarray | list[ndarray]) -> ndarray | list[ndarray]:
        """calculate total volumes.

        Assume that N=len(dfs), m=obs_num, m*n=len(df), then obs_arrs.shape = (N, n)

        Args:
            volume_arrs (ndarray | list[ndarray]): _description_

        Returns:
            ndarray | list[ndarray]: _description_
        """
        if isinstance(volume_arrs, list):
            total_volume_arrs: list[ndarray] = []
            for volume_arr in volume_arrs:
                total_volume_arr: ndarray = np.sum(volume_arr, axis=1)
                assert len(total_volume_arr.shape) == 1
                total_volume_arrs.append(total_volume_arr)
        elif isinstance(volume_arrs, ndarray):
            assert len(volume_arrs.shape) == 3
            total_volume_arrs: ndarray = np.sum(volume_arrs, axis=2)
        else:
            raise NotImplementedError
        return total_volume_arrs

    def _count_cumsum_datas(self) -> ndarray:
        if isinstance(self.rv_arrs, ndarray):
            num_data_per_series: int = self.rv_arrs.shape[1] - self.input_time_length - 1
            num_data_arr: ndarray = np.cumsum(
                np.ones(self.rv_arrs.shape[0]) * num_data_per_series
            ) - 1
        elif isinstance(self.rv_arrs, list):
            num_datas: list[int] = []
            for rv_arr in self.rv_arrs:
                num_datas.append(len(rv_arr) - self.input_time_length)
            num_data_arr: ndarray = np.cumsum(
                np.array(num_datas)
            ) - 1
        else:
            raise NotImplementedError
        return num_data_arr

    def _normalize(self, feature_arrs: ndarray) -> ndarray | list[ndarray]:
        """normalize features.

        Args:
            feature_arrs (ndarray | list[ndarray]): _description_

        Returns:
            ndarray | list[ndarray]: _description_
        """
        if isinstance(feature_arrs, ndarray):
            feature_arrs = (
                feature_arrs - feature_arrs.mean()
            ) / (feature_arrs.std() + 1e-10)
        else:
            raise NotImplementedError
        return feature_arrs

    def plot_features(self, img_name: str) -> None:
        fig = plt.figure(figsize=(30,20), dpi=50, facecolor="w")
        ax1: Axes = fig.add_subplot(3,1,1)
        self._hist_features(
            ax1, self.return_arrs, 30, xlabel="return", title=""
        )
        ax2: Axes = fig.add_subplot(3,1,2)
        self._hist_features(
            ax2, self.volume_arrs, 30, xlabel="volume", title=""
        )
        ax3: Axes = fig.add_subplot(3,1,3)
        self._hist_features(
            ax3, self.rv_arrs, 30, xlabel="realized volatility", title=""
        )
        fig_save_path: Path = self.imgs_path / img_name
        plt.savefig(str(fig_save_path), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    def _hist_features(
        self,
        ax: Axes,
        feature_arrs: ndarray,
        num_bins: int,
        xlabel: str,
        title: str
    ) -> None:
        mean_arr: float = np.mean(feature_arrs)
        std_arr: float = np.std(feature_arrs)
        min_arr: float = np.min(feature_arrs)
        max_arr: float = np.max(feature_arrs)
        ax.hist(
            feature_arrs.flatten(), num_bins,
            label=f"mean={mean_arr:.2f} std={std_arr:.2f} " +
            f"min={min_arr:.2f} max={max_arr:.2f}"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frequency")
        ax.set_title(title)
        ax.legend()

    def __len__(self):
        return int(self.num_data_arr[-1]) + 1

    def __getitem__(self, index) -> tuple[Tensor]:
        idx1: int = bisect.bisect_left(self.num_data_arr, index)
        if idx1 == 0:
            idx2: int = index
        else:
            idx2: int = int(index - self.num_data_arr[idx1-1] - 1)
        if isinstance(self.return_arrs, ndarray):
            return_tensor: Tensor = torch.from_numpy(
                self.return_arrs[
                    idx1, idx2:idx2+self.input_time_length
                ].astype(np.float32)
            ).view(1,-1)
        if isinstance(self.volume_arrs, ndarray):
            volume_tensor: Tensor = torch.from_numpy(
                self.volume_arrs[
                    idx1, idx2:idx2+self.input_time_length
                ].astype(np.float32)
            ).view(1,-1)
        if isinstance(self.rv_arrs, ndarray):
            rv_tensor: Tensor = torch.from_numpy(
                self.rv_arrs[
                    idx1, idx2:idx2+self.input_time_length
                ]
            ).view(1,-1)
            target_tensor: Tensor = torch.Tensor(
                [self.rv_arrs[idx1, idx2+self.input_time_length+1]]
            )
        input_tensor = torch.cat(
            [return_tensor, volume_tensor, rv_tensor]
        )
        return input_tensor, target_tensor

curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
root_path: pathlib.Path = curr_path.parents[2]
imgs_path = root_path / "imgs"
datas_path = root_path / "datas"
artificial_datas_path = datas_path / "artificial_datas" / "intraday" / "afcn" / "random"
aapl_datas_path = datas_path / "real_datas" / "intraday" / "aapl"
sp_datas_path = datas_path / "real_datas" / "intraday" / "sp500"
rv_dataset = RVDataset(
    artificial_datas_path, None, 30, 10, imgs_path
)
rv_dataset.plot_features("features_afcn_random.pdf")
"""
rv_dataset = RVDataset(
    aapl_datas_path, ["AAPL2018.csv"], 10, 10, imgs_path
)
rv_dataset.plot_features("features_aapl_2018.pdf")
"""