import bisect
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

class RVDataset(Dataset):
    def __init__(
        self,
        olhcv_path: Path,
        obs_num: int,
        input_time_length: int,
        imgs_path: Optional[Path] = None
    ) -> None:
        """initialization.

        Create Dataset for realized volatility forecasting.
        initialization method create dataset by following procedure.
            1. read all csv files from olhcv_path folder.
            2. fold the time series to calculate realized volatility.
                Assume that length of a csv being time_series_length, then folded array is
                (time_series_length/obs_num, obs_num). Note that time_series_length should be
                divisible by obs_num.
            3. calculate return, volume, realized volatility.
                shape of return_arrs, volume_arrs and rv_arrs is (data_num, time_series_length/obs_num)

        Args:
            olhcv_path (Path): _description_
            obs_num (int): _description_
        """
        olhcv_dfs = self._read_csvs(olhcv_path)
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
        if imgs_path is not None:
            feature_img_path: Path = imgs_path / "feature.pdf"
            self.plot_features(feature_img_path)

    def _read_csvs(
        self,
        csvs_path: Path,
    ) -> list[DataFrame]:
        """read all csv files in given folder path.
        """
        dfs: list[DataFrame] = []
        for csv_path in sorted(csvs_path.iterdir()):
            if csv_path.suffix == ".csv":
                dfs.append(pd.read_csv(csv_path, index_col=0))
        return dfs

    def _fold_dfs(self, dfs: DataFrame, colname: str) -> ndarray | list[ndarray]:
        """fold specific columns of dfs.

        Assume that N=len(dfs), m=obs_num, m*n=len(df), then
            folded_columns.shape = (N, n, m)

        Args:
            dfs (DataFrame): _description_
            colname (str): _description_

        Returns:
            ndarray | list[ndarray]: _description_
        """
        folded_columns: list[ndarray] = []
        for df in dfs:
            column_arr: ndarray = df[colname].values[:,np.newaxis]
            assert len(column_arr.shape) == 2
            if len(column_arr) % self.obs_num != 0:
                warnings.warn(
                    f"number of obs {len(column_arr)} cannot be devised by {self.obs_num}. " +
                    f"data will be pruned to {np.floor(len(column_arr) / self.obs_num)}."
                )
                column_arr = column_arr[np.floor(len(column_arr) / self.obs_num)]
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

    def plot_features(self, fig_save_path: Path) -> None:
        pass

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

curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
olhcv_path = root_path / "datas" / "artificial_datas" / "intraday" / "afcn" / "random"
rv_dataset = RVDataset(olhcv_path, 10, 10)
print(rv_dataset.__getitem__(1499))