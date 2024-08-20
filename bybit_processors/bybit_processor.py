from datetime import datetime
from datetime import timedelta
import gzip
import pandas as pd
from pandas import DataFrame
import pathlib
from pathlib import Path
from rich import print
from typing import Optional
import urllib.request as request
import warnings

class BybitProcessor:
    """BybitProcessor class.
    
    BybitProcessor provides preprocessed csv data from the Bybit exchange.
    Bybit (https://public.bybit.com/trading/) provides the historical data with following format.

    |timestamp    |symbol |side  |size  |price   |tickDirection |trdMatchID |\
    |1.609546e+09 |BTCUSD |Sell  |23099 |29394.5 |ZeroMinusTick |           |\
    
    |grossValue |homeNotional |foreignNotional |
    |78582728.0 |23099        |0.785827        |

    BybitProcessor creates the following structured dataframe from
    
    | time           | session_id |event_flag | event_volume | event_price | market_price |
    | 00:00:00.50400 | 1          |execution  | 22000        | 275         | 275          |
    ...
    """
    def __init__(
        self,
        csv_datas_path: Optional[Path] = None
    ) -> None:
        """initialization.
        
        Args:
            csv_datas_path (Optional[Path]): Path to the csv data to be saved.
                Ex) datas/real_datas/bybit_csv
        """
        self.csv_datas_path: Optional[Path] = csv_datas_path

    def download_datas_from_bybit(
        self,
        tickers: list[str],
        start_date: int | str = 20150101,
        end_date: int | str = 20211231
    ) -> None:
        """Download intraday datas.

        Heres are the list of tickers and the corresponding data period.
            - ADAUSD: 20220325 ~
            - BITUSD: 20211115 ~
            - BTCUSD: 20191001 ~
            - DOTUSD: 20211013 ~
            - EOSUSD: 20191001 ~
            - ETHUSD: 20191001 ~
            - LTCUSD: 20220331 ~
            - LUNAUSD:20220331 ~
            - MANAUSD:20220331 ~
            - SOLUSD: 20220325 ~
            - XRPUSD: 20191001 ~
        
        Args:
            tickers (list[str]): list of tickers.
            start_date (int | str): Start date with %Y%m%d format.
            end_date (int | str): End date with %Y%m%d format.
        """
        assert self.csv_datas_path is not None
        if not self.csv_datas_path.exists():
            self.csv_datas_path.mkdir(parents=True)
        current_datetime = datetime.strptime(str(start_date), "%Y%m%d")
        end_datetime = datetime.strptime(str(end_date), "%Y%m%d")
        print("Start downloading datas from Bybit.")
        print(f"tickers: {tickers}")
        print(f"start_date: {start_date} end_date: {end_date}")
        while current_datetime <= end_datetime:
            current_date_str: str = current_datetime.strftime("%Y-%m-%d")
            current_date_int: int = int(current_datetime.strftime("%Y%m%d"))
            current_date_datas_path: Path = self.csv_datas_path / f"{current_date_int}"
            if not current_date_datas_path.exists():
                current_date_datas_path.mkdir(parents=True)
            for ticker in tickers:
                current_date_df: Optional[DataFrame] = self._download_data_from_bybit(
                    date_str=current_date_str, ticker=ticker
                )
                if current_date_df is not None:
                    save_path: Path = current_date_datas_path / f"Bybit{ticker}_{current_date_int}.csv"
                    current_date_df.to_csv(str(save_path))
            current_datetime += timedelta(days=1)
        print("Done!")

    def _download_data_from_bybit(
        self,
        date_str: str,
        ticker: str
    ) -> Optional[DataFrame]:
        """Download intraday data from Bybit.
        
        Args:
            date_str (str): Date with %Y-%m-%d format.
            ticker (str): Ticker.
        """
        url_to_data: str = f"https://public.bybit.com/trading/{ticker}/{ticker}{date_str}.csv.gz"
        temp_path: Path = pathlib.Path("temp")
        try:
            request.urlretrieve(url_to_data, temp_path)
            with gzip.open(temp_path, "rt") as f:
                df = pd.read_csv(f)
            temp_path.unlink()
        except Exception as e:
            return None
        df.sort_values(by="timestamp", ascending=True, inplace=True)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="s", format="%Y-%m-%d %H:%M:%S.%f"
        ).dt.time
        df.rename(columns={"timestamp": "time"}, inplace=True)
        df.set_index("time", inplace=True)
        df["session_id"] = 1
        df["event_flag"] = "execution"
        df.rename(columns={"size": "event_volume"}, inplace=True)
        df.rename(columns={"price": "event_price (avg)"}, inplace=True)
        df["market_price"] = df["event_price (avg)"]
        df = df.loc[
            :,["session_id", "event_flag", "event_volume", "event_price (avg)", "market_price"]
        ]
        return df
