import csv
import json
from pathlib import Path
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
        txt_datas_path: Path,
        csv_datas_path: Path,
        quote_num: int = 10,
        is_execution_only: bool = True
    ) -> None:
        """initialization.

        FlexProcessor scan all txt files under txt_datas_path and create the same directory structure
        as txt_datas_path under csv_datas_path with csv files.

        Args:
            txt_datas_path (Path): target folder path.
            csv_datas_path (Path): folder path to create csv datas.
            quote_num (int): number of limit prices to store in order books.
            is_execution_only (bool): wether to record execution event only. default to True.
        """
        self.txt_datas_path: Path = txt_datas_path
        assert self.txt_datas_path.exists()
        self.csv_datas_path: Path = csv_datas_path
        self.quote_num: int = quote_num
        self.is_execution_only: bool = is_execution_only
        self.column_names: list[str] = self._create_columns()

    def _create_columns(self) -> list[str]:
        column_names: list[str] = [
            "time", "event_flag", "market_price", "best_buy", "best_sell"
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

    def convert_all_txt2csv(self) -> None:
        for txt_path in self.txt_datas_path.rglob("*.txt"):
            csv_path: Path = \
                self.csv_datas_path / txt_path.relative_to(
                    self.txt_datas_path
                ).with_suffix(".txt")
            self.convert_txt2csv(
                txt_path, csv_path
            )

    def convert_txt2csv(
        self,
        txt_path: Path,
        csv_path: Path
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
                    log_dic["Data"]["market_price"],
                    log_dic["Data"]["mid_price"],
                    log_dic["Data"]["best_bid"],
                    log_dic["Data"]["best_ask"]
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
        event_price: Optional[float] = None
        for message_dic in message_dics:
            if message_dic["tag"] == "1P":
                if event_price is None:
                    event_price = float(message_dic["price"])
                else:
                    if event_price != float(message_dic["price"]):
                        raise ValueError(
                            f"executions with different prices found! {event_price} and {float(message_dic["price"])}"
                        )
            elif message_dic["tag"] == "VL":
                event_volume += int(message_dic["volume"])
            else:
                pass
        if event_price is not None or 0 < event_volume:
            if event_price is None or event_volume == 0:
                raise ValueError(
                    f"incompatible log: event_price={event_price}, event_volume={event_volume}"
                )
            else:
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
        for i, price, volume in enumerate(order_book_dic.items()):
            if i+1 <= self.quote_num:
                price_volumes.append(float(price))
                price_volumes.append(int(volume))
            else:
                break
        while len(price_volumes) < 2*self.quote_num:
            price_volumes.append(None)
        assert len(price_volumes) == 2*self.quote_num
        return price_volumes
