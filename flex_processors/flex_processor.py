import json
import pathlib
from pathlib import Path

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

    datetime       | event_flag | market_price | mid_price | best_buy | best_sell | ...
    09:00:00.50400 | execution  | 275          | 275.5     | 275      | 276       | ...
    ...
    """
    def __init__(
        self,
        txt_datas_path: Path,
        csv_datas_path: Path,
        is_execution_only: bool = True
    ) -> None:
        """initialization.

        FlexProcessor scan all txt files under txt_datas_path and create the same directory structure
        as txt_datas_path under csv_datas_path with csv files.

        Args:
            txt_datas_path (Path): target folder path.
            csv_datas_path (Path): folder path to create csv datas.
            is_execution_only (bool): wether to record execution event only. default to True.
        """
        self.txt_datas_path: Path = txt_datas_path
        assert self.txt_datas_path.exists()
        self.csv_datas_path: Path = csv_datas_path
        self.is_execution_only: bool = is_execution_only

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
