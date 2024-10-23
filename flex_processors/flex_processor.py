import csv
from datetime import datetime
from datetime import timedelta
import json
import pandas as pd
from pandas import Timestamp
import pathlib
from pathlib import Path
import subprocess
from tqdm import tqdm
from typing import Optional
import warnings

class FlexProcessor:
    """FlexProcessor class.

    FlexProcessor provide preprocessed csv data from txt data with FLEX format.
    FLEX historical dataset is time series data of all indivisual orders and executions
    by Japan Exchange Group (JPX). FLEX data is provided as following format, for example.

    {
        "Data": {
            "time":"09:00:00.307000", # time that the events occur.
            "code":"1301", # ticker code.
            "status":"", # ?
            "message":[ # message that describe the events
                {"tag":"QB","price":"272","qty":16000->15000"},
                    # the volume at price 272 in buy order book decreased from 16000 to 15000.
                {"tag":"QS","proce":"277","qty":"6000->5000"},
                    # the volume at price 277 in sell order book ed from 6000 to 5000.
            ],
            "market_price":"NaN",
            "best_ask":"222",
            "best_bid":"278",
            "mid_price":"250",
            "buy_book":{"MO":"13000","278":"2000","277":"1000","276":"5000",...},
            "sell_book":{"MO":"16000","222","2000","275":"4000","276":"13000",...}
        }
    }\n
    {
        "Data": {
            "time":"09:00:00.504000",
            "code":"1301",
            "status":"",
            "message":[
                {"tag":"1P","price":"275"},
                {"tag":"VL","volume":"22000"},
                    # execution with price 275, volume 22000.
                {"tag":"QS","price":"MO","qty":"16000->0"},
                {"tag":"QS","price":"222","qty":"2000->0"},
                {"tag":"QS","price":"275","qty":"4000->0"},
                {"tag":"QS","price":"277","qty":"5000->1000"},
                {"tag":"QB","price":"MO","qty":"13000->0"},
                {"tag":"QB","price":"278","qty":"2000->0"},
                {"tag":"QB","price":"277","qty":"1000->0"},
                {"tag":"QB","price":"276","qty":"5000->0"},
                {"tag":"QB","price":"275","qty":"18000->17000"},
                {"tag":"QB","price":"272","qty":"15000->11000"}
            ],
            "market_price":"275",
            "best_ask":"276",
            "best_bid":"275",
            "mid_price":"275.5",
            "buy_book":{"275":"17000","274":"19000","273":"24000","272":"11000",...},
            "sell_book":{"276":"13000","277":"1000","278":"5000","279":"4000",...}
        }
    }

    Flex Processor create following structured dataframe from the above txt data with FLEX format, for example.

    | time           | session_id |event_flag | event_volume | event_price | market_price | mid_price | \
    | 09:00:00.50400 | 1          |execution  | 22000        | 275         | 275          | 275.5     | \

    | best_buy | best_sell | buy1_price | buy1_volume | buy2_price | buy2_volume | ...
    | 275      | 276       | 275        | 17000       | 274        | 19000       | ...
    """
    def __init__(
        self,
        txt_datas_path: Optional[Path] = None,
        csv_datas_path: Optional[Path] = None,
        flex_downloader_path: Optional[Path] = None,
        quote_num: int = 10,
        is_execution_only: bool = True,
        is_mood_aware: bool = False,
        is_wc_rate_aware: bool = False,
        session1_end_time_str: str = "11:30:00.000000",
        session2_start_time_str: str = "12:30:00.000000"
    ) -> None:
        """initialization.

        FlexProcessor scan all txt files under txt_datas_path and create the same directory structure
        as txt_datas_path under csv_datas_path with csv files.

        There are 2 sessions in Japan's stock market in 1 day.

        Args:
            txt_datas_path (Path): target folder path. default to None.
            csv_datas_path (Path): folder path to create csv datas. default to None.
            flex_downloader_path (Path): path to FlexDownloader. default to None.
                Ex) ~/flex_full_processed_downloader/downloader.rb
            quote_num (int): number of limit prices to store in order books. default to 10.
            is_execution_only (bool): wether to record execution event only. default to True.
        """
        self.txt_datas_path: Optional[Path] = txt_datas_path
        self.csv_datas_path: Optional[Path] = csv_datas_path
        self.flex_downloader_path: Optional[Path] = flex_downloader_path
        self.quote_num: int = quote_num
        self.is_execution_only: bool = is_execution_only
        self.is_mood_aware: bool = is_mood_aware
        self.is_wc_rate_aware: bool = is_wc_rate_aware
        self.session1_end_time: Timestamp = pd.to_datetime(session1_end_time_str).time()
        self.session2_start_time: Timestamp = pd.to_datetime(session2_start_time_str).time()

    def download_datas(
        self,
        tickers: str,
        start_date: int | str = 20150101,
        end_date: int | str = 20211231
    ) -> None:
        assert self.txt_datas_path is not None
        if not self.txt_datas_path.exists():
            self.txt_datas_path.mkdir(parents=True)
        assert self.flex_downloader_path is not None
        assert self.flex_downloader_path.suffix == ".rb"
        current_datetime = datetime.strptime(str(start_date), "%Y%m%d")
        end_datetime = datetime.strptime(str(end_date), "%Y%m%d")
        while current_datetime <= end_datetime:
            current_date = int(current_datetime.strftime("%Y%m%d"))
            datas_path: Path = pathlib.Path(__file__).resolve().parents[0] / str(current_date)
            download_command: str = f"ruby {str(self.flex_downloader_path)} {current_date} {tickers}"
            try:
                _ = subprocess.run(download_command, shell=True)
                for data_path in datas_path.iterdir():
                    if data_path.suffix == ".txt":
                        destination_folder_path: Path = self.txt_datas_path / str(current_date)
                        if not destination_folder_path.exists():
                            destination_folder_path.mkdir(parents=True)
                        destination_path: Path = destination_folder_path / data_path.name
                        move_command: str = f"mv {str(data_path)} {str(destination_path)}"
                        _ = subprocess.run(move_command, shell=True)
                remove_command: str = f"rm -rf {str(datas_path)}"
                _ = subprocess.run(remove_command, shell=True)
            except Exception as e:
                print(e)
            current_datetime += timedelta(days=1)
            self.convert_all_txt2csv(is_bybit_format=False, is_display_path=False)
        self.txt_datas_path.unlink()

    def _create_columns(self, is_bybit_format: bool = False) -> list[str]:
        column_names: list[str] = [
            "time", "session_id", "event_flag", "event_volume", "event_price (avg)", "market_price",
            "mid_price", "best_buy", "best_sell"
        ]
        if not is_bybit_format:
            [
                (column_names.append(f"buy{i+1}_price"), column_names.append(f"buy{i+1}_volume")) \
                    for i in range(self.quote_num)
            ]
            [
                (column_names.append(f"sell{i+1}_price"), column_names.append(f"sell{i+1}_volume")) \
                    for i in range(self.quote_num)
            ]
        if self.is_mood_aware:
            column_names.append("mood")
        if self.is_wc_rate_aware:
            column_names.append("wc_rate")
            column_names.append("time_window_size")
        return column_names

    def convert_all_txt2csv(
        self,
        is_bybit_format: bool,
        is_display_path: bool = True,
    ) -> None:
        for txt_path in tqdm(self.txt_datas_path.rglob("*.txt")):
            csv_path: Path = \
                self.csv_datas_path / txt_path.relative_to(
                    self.txt_datas_path
                ).with_suffix(".csv")
            csv_parent_path: Path = csv_path.parent
            if not csv_parent_path.exists():
                csv_parent_path.mkdir(parents=True)
            self.convert_txt2csv(
                txt_path, csv_path,
                is_bybit_format=is_bybit_format,
                is_display_path=is_display_path
            )
            txt_path.unlink()

    def convert_txt2csv(
        self,
        txt_path: Path,
        csv_path: Path,
        is_display_path: bool,
        is_bybit_format: bool = False
    ) -> None:
        assert txt_path.suffix == ".txt"
        assert csv_path.suffix == ".csv"
        column_names: list[str] = self._create_columns(is_bybit_format=is_bybit_format)
        with open(txt_path, mode="r") as f:
            with open(csv_path, mode="w") as g:
                writer = csv.writer(g)
                writer.writerow(column_names)
                line: str
                if is_display_path:
                    print(f"convert from {str(txt_path)} to {str(csv_path)}")
                for line in f.read().splitlines():
                    try:
                        log_dic: dict[
                            str, dict[str, dict[str, list | dict, str]]
                        ] = json.loads(line.replace("'", '"'))
                        log_columns: list[str] = self._extract_info_from_log(log_dic)
                        log_columns = self._add_mood(log_dic, log_columns)
                        log_columns = self._add_wc_rate(log_dic, log_columns)
                    except Exception as e:
                        print(e)
                        print(line)
                        continue
                    if len(log_columns) == 0:
                        pass
                    elif len(log_columns) == len(column_names):
                        writer.writerow(log_columns)
                    else:
                        print(len(column_names), len(log_columns))
                        raise ValueError(
                            "column mismatch.\n" +
                            f"column names = {column_names}\n" +
                            f"records = {log_columns}"
                        )

    def _extract_info_from_log(
        self,
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]],
        is_bybit_format: bool = False
    ) -> list[str]:
        message_dics: list[dict[str, str]] = log_dic["Data"]["message"]
        if self.is_execution_only:
            execution_infos: list[str | float | int] = \
                self._extract_execution_info_from_message_dics(message_dics)
        else:
            raise NotImplementedError
        if len(execution_infos) == 0:
            return []
        elif len(execution_infos) == 3:
            time_str: str = log_dic["Data"]["time"]
            log_columns: list[str] = [time_str]
            if "session_id" in log_dic["Data"].keys():
                log_columns.append(log_dic["Data"]["session_id"])
            elif is_bybit_format:
                log_columns.append("1")
            else:
                t: Timestamp = pd.to_datetime(time_str).time()
                if t <= self.session1_end_time:
                    session_id: str = "1"
                elif self.session2_start_time <= t:
                    session_id: str = "2"
                else:
                    raise ValueError(f"cannot identify session. time={time_str}")
                log_columns.append(session_id)
            log_columns.extend(execution_infos)
            log_columns.extend(
                [
                    float(log_dic["Data"]["market_price"]),
                    float(log_dic["Data"]["mid_price"]),
                    float(log_dic["Data"]["best_bid"]),
                    float(log_dic["Data"]["best_ask"])
                ]
            )
        else:
            raise ValueError(
                f"length of execution_infos must be 3 (event_flag, event_volume, event_price) " +
                f"but found {execution_infos}"
            )
        if is_bybit_format:
            return log_columns
        buy_price_volumes: list[Optional[int | float]] = self._extract_price_volume_info_from_log(
            log_dic, key_name="buy_book"
        )
        log_columns.extend(buy_price_volumes)
        sell_price_volumes: list[Optional[int | float]] = self._extract_price_volume_info_from_log(
            log_dic, key_name="sell_book"
        )
        log_columns.extend(sell_price_volumes)
        return log_columns

    def _extract_execution_info_from_message_dics(
        self,
        message_dics: list[dict[str, str]]
    ) -> list[int | float]:
        """
        Args:
            message_dics (list[dict[str, str]]): ex: [
                {"tag":"1P","price":"275"},
                {"tag":"VL","volume":"22000"},
                {"tag":"QS","price":"MO","qty":"16000->0"},
                {"tag":"QS","price":"222","qty":"2000->0"},
                ...
            ]

        Returns:
            list[str]: [] or [event_flag, event_volume, event_price]
        """
        execution_infos: list[str] = []
        event_flag: str = "execution"
        event_volume: int = 0
        event_prices: list[float] = []
        for message_dic in message_dics:
            if message_dic["tag"] == "1P":
                event_prices.append(float(message_dic["price"]))
            elif message_dic["tag"] == "VL":
                event_volume += int(message_dic["volume"])
            else:
                pass
        if 0 < len(event_prices):
            event_price: float = sum(event_prices) / len(event_prices)
            execution_infos = [
                event_flag, event_volume, event_price
            ]
        return execution_infos

    def _extract_price_volume_info_from_log(
        self,
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]],
        key_name: str
    ) -> list[Optional[int | float]]:
        price_volumes: list[str] = []
        order_book_dic: dict[str, str] = log_dic["Data"][key_name]
        for i, (price, volume) in enumerate(order_book_dic.items()):
            if i+1 <= self.quote_num:
                if price == "MO":
                    price_volumes.append(None)
                else:
                    price_volumes.append(float(price))
                price_volumes.append(int(volume))
            else:
                break
        while len(price_volumes) < 2*self.quote_num:
            price_volumes.append(None)
        assert len(price_volumes) == 2*self.quote_num
        return price_volumes
    
    def _add_mood(
        self,
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]],
        log_columns: list[str]
    ) -> list[str]:
        if self.is_mood_aware:
            mood: float = float(log_dic["Data"]["mood"])
            log_columns.append(mood)
        else:
            if "mood" in log_dic["Data"]:
                warnings.warn(
                    "set not to write mood even though mood is recorded in simulation log."
                )
        return log_columns
    
    def _add_wc_rate(
        self,
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]],
        log_columns: list[str]
    ) -> list[str]:
        if self.is_wc_rate_aware:
            wc_rate: float = float(log_dic["Data"]["wc_rate"])
            log_columns.append(wc_rate)
            time_window_size: int = int(log_dic["Data"]["time_window_size"])
            log_columns.append(time_window_size)
        else:
            if "wc_rate" in log_dic["Data"]:
                warnings.warn(
                    "set not to write wc_rate even though wc_rate is recorded in simulation log."
                )
            if "time_window_size" in log_dic["Data"]:
                warnings.warn(
                    "set not to write time_window_size even though time_window_size is recorded in simulation log."
                )
        return log_columns
