import csv
from datetime import datetime
from datetime import timedelta
import json
from pathlib import Path
import subprocess
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

    | time           | event_flag | event_volume | event_price | market_price | mid_price |\n
    | 09:00:00.50400 | execution  | 22000        | 275         | 275          | 275.5     |\n

    | best_buy | best_sell | buy1_price | buy1_volume | buy2_price | buy2_volume | ...
    | 275      | 276       | 275        | 17000       | 274        | 19000       | ...
    """
    def __init__(
        self,
        txt_datas_path: Optional[Path] = None,
        csv_datas_path: Optional[Path] = None,
        flex_downloader_path: Optional[Path] = None,
        quote_num: int = 10,
        is_execution_only: bool = True
    ) -> None:
        """initialization.

        FlexProcessor scan all txt files under txt_datas_path and create the same directory structure
        as txt_datas_path under csv_datas_path with csv files.

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
        self.column_names: list[str] = self._create_columns()

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
        subprocess.run(f"cd {str(self.txt_datas_path.resolve())}", shell=True)
        while current_datetime <= end_datetime:
            current_date = int(current_datetime.strftime("%Y%m%d"))
            command: str = f"ruby {str(self.flex_downloader_path)} {current_date} {tickers}"
            try:
                subprocess.run(command, shell=True)
            except:
                pass
            current_datetime += timedelta(days=1)

    def _create_columns(self) -> list[str]:
        column_names: list[str] = [
            "time", "event_flag", "event_volume", "event_price (avg)",
            "market_price", "mid_price", "best_buy", "best_sell"
        ]
        [
            (column_names.append(f"buy{i+1}_price"), column_names.append(f"buy{i+1}_volume")) \
                for i in range(self.quote_num)
        ]
        [
            (column_names.append(f"sell{i+1}_price"), column_names.append(f"sell{i+1}_volume")) \
                for i in range(self.quote_num)
        ]
        return column_names

    def convert_all_txt2csv(self, is_display_path: bool = True) -> None:
        for txt_path in self.txt_datas_path.rglob("*.txt"):
            csv_path: Path = \
                self.csv_datas_path / txt_path.relative_to(
                    self.txt_datas_path
                ).with_suffix(".csv")
            csv_parent_path: Path = csv_path.parent
            if not csv_parent_path.exists():
                csv_parent_path.mkdir(parents=True)
            self.convert_txt2csv(
                txt_path, csv_path,
                is_display_path
            )

    def convert_txt2csv(
        self,
        txt_path: Path,
        csv_path: Path,
        is_display_path: bool
    ) -> None:
        assert txt_path.suffix == ".txt"
        assert csv_path.suffix == ".csv"
        if csv_path.exists():
            warnings.warn(
                f"file: {str(csv_path)} already exists. " +
                "the content of the file will be overwritten."
            )
        with open(txt_path, mode="r") as f:
            with open(csv_path, mode="w") as g:
                writer = csv.writer(g)
                writer.writerow(self.column_names)
                line: str
                if is_display_path:
                    print(f"convert from {str(txt_path)} to {str(csv_path)}")
                for line in f.read().splitlines():
                    log_dic: dict[
                        str, dict[str, dict[str, list | dict, str]]
                    ] = json.loads(line.replace("'", '"'))
                    log_columns: list[str] = self._extract_info_from_log(log_dic)
                    if len(log_columns) == 0:
                        pass
                    elif len(log_columns) == len(self.column_names):
                        writer.writerow(log_columns)
                    else:
                        print(len(self.column_names), len(log_columns))
                        raise ValueError(
                            "column mismatch.\n" +
                            f"column names = {self.column_names}\n" +
                            f"records = {log_columns}"
                        )

    def _extract_info_from_log(
        self,
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]]
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
            log_columns: list[str] = [log_dic["Data"]["time"]]
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
