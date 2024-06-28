from datetime import date
from datetime import datetime
from datetime import timedelta
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional
import warnings

class OHLCVProcessor:
    """OHLCV Processor class."""

    def __init__(
        self,
        tickers: list[int],
        daily_ohlcv_dfs_path: Path,
        all_time_ohlcv_dfs_path: Optional[Path],
        start_date: date,
        end_date: date,
    ) -> None:
        """initialization."""
        self.tickers: list[int] = tickers
        self.daily_ohlcv_dfs_path: Path = daily_ohlcv_dfs_path
        self.all_time_ohlcv_dfs_path: Optional[Path] = all_time_ohlcv_dfs_path
        self.start_date: date = start_date
        self.end_date: date = end_date

    def concat_all_ohlcv_dfs(self, ticker_first: bool = True) -> None:
        """concat all OHLCV dataframes of given tickers.

        concat all dataframes within daily_ohlcv_dfs_path to files as same number as number of tickers.
        dataframes of specified tickers [ticker1, ticker2, ...] will be converted to files and saved as:
        
        all_time_ohlcv_dfs_path
            |- {ticker1}_{start_date}_{end_date}.csv
            |- {ticker2}_{start_date}_{end_date}.csv
            |- ...
        """
        assert self.all_time_ohlcv_dfs_path is not None
        for ticker in self.tickers:
            start_date_str: str = self.start_date.strftime(format='%Y%m%d')
            end_date_str: str = self.end_date.strftime(format='%Y%m%d')
            all_time_ohlcv_df_path: Path = (
                self.all_time_ohlcv_dfs_path
                / f"{ticker}_{start_date_str}_{end_date_str}.csv"
            )
            if ticker_first:
                daily_ohlcv_dfs_path: Path = self.daily_ohlcv_dfs_path / ticker
            else:
                daily_ohlcv_dfs_path: Path = self.daily_ohlcv_dfs_path
            _ = self.concat_ohlcv_dfs(
                daily_ohlcv_dfs_path,
                specific_name=ticker,
                all_time_ohlcv_df_path=all_time_ohlcv_df_path,
                start_date=self.start_date,
                end_date=self.end_date,
            )

    def concat_ohlcv_dfs(
        self,
        daily_ohlcv_dfs_path: Path,
        specific_name: Optional[str],
        all_time_ohlcv_df_path: Optional[Path],
        start_date: date,
        end_date: date,
    ) -> Optional[DataFrame]:
        """concat OHLCV dataframes saved day by day.

        concat all dataframes that contains specific_name and save to all_time_ohlcv_df_path as 1 file.
        Assume that structure of daily_ohlcv_dfs_path is as follows, for example.

        daily_ohlcv_dfs_path
            |- 20150107
            |   |- Full9202_20160107.csv
            |- 20150108
            |   |- Full9202_20160108.csv
            |- ...

        This method sequencially searchs dataframe at date between start_date and end_date.
        To concat all intraday csv data of ticker 9202 between 2015/01/05 and 2021/12/31
        to all_time_ohlcv_df_path, specify specific_name="9202", start_date=date(2015,1,5), end_date=date(2021,12,31).

        Args:
            daily_ohlcv_dfs_path (Path): folder path whose structure is described above.
            specific_name (str, optional): If not None, choose file whose name contains specific_name.
            all_time_ohlcv_df_path (Path, optional): If not None, save concatted dataframe to all_time_ohlcv_df_path.
            start_date (date): start date.
            end_date (date): end date.
        """
        all_time_df: Optional[DataFrame] = None
        today_date: date = start_date
        while True:
            today_str: str = today_date.strftime(format="%Y%m%d")
            for today_df_path in daily_ohlcv_dfs_path.rglob("*.csv"):
                file_name: str = today_df_path.name
                if (
                    today_str in file_name and
                    specific_name in file_name
                ):
                    today_df: DataFrame = pd.read_csv(today_df_path, index_col=0)
                    today_df.index = pd.to_datetime(today_df.index, format="%H:%M:%S")
                    today_df.index = today_df.index.time
                    all_time_df: DataFrame = self._concat_ohlcv_dfs(
                        today_df, today_date=today_date, all_time_df=all_time_df
                    )
                    break
            today_date += timedelta(days=1)
            if end_date < today_date:
                break
        if all_time_ohlcv_df_path is not None:
            all_time_ohlcv_df_path = all_time_ohlcv_df_path.with_suffix(".csv")
            if all_time_df is None:
                warnings.warn(
                    f"could not find any csv files to contain specific_name within {daily_ohlcv_dfs_path}."
                )
            else:
                all_time_df.to_csv(str(all_time_ohlcv_df_path))
        return all_time_df

    def _concat_ohlcv_dfs(
        self,
        today_df: DataFrame,
        today_date: date,
        all_time_df: Optional[DataFrame],
    ) -> DataFrame:
        datetimes = [datetime.combine(today_date, t) for t in today_df.index]
        today_df.index = datetimes
        if all_time_df is None:
            all_time_df: DataFrame = today_df
        else:
            all_time_df = pd.concat([all_time_df, today_df], axis=0)
        return all_time_df
