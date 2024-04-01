from pams.logs import Logger
from typing import List, TypeVar
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.logs.base import Log
from pams.market import Market
from pams.order_book import OrderBook

MarketID = TypeVar("MarketID")

class FlexLogger(Logger):
    """FlexLogger class.

    FlexLogger save logs of pams simulation by FLEX format.
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
    """
    def __init__(self) -> None:
        super().__init__()
        self.times_dic: dict[MarketID, int] = {}
        self.logs_dic: dict[MarketID, list[dict[str, list | dict | str]]] = {}

    def process_order_log(self, log: OrderLog) -> None:
        order_time: int = log.time
        market_id: Market = log.market_id

    def process_cancel_log(self, log: CancelLog) -> None:
        cancel_time: int = log.cancel_time
        market_id: Market = log.market_id

    def process_execution_log(self, log: ExecutionLog) -> None:
        execution_time: int = log.time
        market_id: Market = log.market_id

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        market: Market = log.market
        market_id: Market = market.market_id
        log_time: int = market.get_time()
        buy_order_book: OrderBook = market.buy_order_book
        sell_order_book: OrderBook =market.sell_order_book

    def _create_empty_log_dic(
        self,
        log_time: int,
        market_id: int
    ) -> dict[str, dict[str, list | dict | str]]:
        empty_log_dic: dict[str, list | dict | str] = {
            "Data": {
                "time": str(log_time),
                "code": str(market_id),
                "status": "",
                "message": [],
                "market_price": "",
                "best_ask": "",
                "best_bid": "",
                "mid_price": "",
                "buy_book": {},
                "sell_book": {}
            }
        }
        return empty_log_dic
