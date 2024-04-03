from pams.logs import Logger
from pams.logs import ExecutionLog
from pams.logs import MarketStepEndLog
from pams.logs.base import MarketStepBeginLog
from pams.market import Market
from pams.order_book import OrderBook
from typing import List, Optional
from typing import TypeVar

MarketID = TypeVar("MarketID")

# TODO: add message of orders. (only execution messages now.)

class FlexSaver(Logger):
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
    """
    def __init__(
        self,
        significant_figures: int = 1
    ) -> None:
        """initialization

        Args:
            significant_figures (int): significant figures to store prices. defaut to 1.
        """
        super().__init__()
        self.current_log_dics: dict[MarketID, dict[str, list | dict | str]] = {}
        self.execution_dics: dict[MarketID, dict[str, int]] = {}
        self.logs_dic: dict[MarketID, list[str]] = {}
        self.previous_buy_price_volume_dic: dict[MarketID, dict[Optional[float], int]] = {}
        self.previous_sell_price_volume_dic: dict[MarketID, dict[Optional[float], int]] = {}
        self.significant_figures: int = significant_figures

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        """process market step begin log.

        store buy/sell order book at the beginning of step t.
        prepare current log at step t.

        Args:
            log (MarketStepBeginLog): market step begin log.
        """
        market: Market = log.market
        market_id: Market = market.market_id
        if market_id not in self.logs_dic.keys():
            self.logs_dic[market_id] = []
        self.execution_dics[market_id] = {}
        previous_buy_order_book: OrderBook = market.buy_order_book
        previous_sell_order_book: OrderBook = market.sell_order_book
        self._add_price_volume_dic(
            market_id, previous_buy_order_book, previous_sell_order_book
        )
        log_time: int = market.get_time()
        current_log_dic: dict[str, dict[str, list | dict | str]] = \
            self._create_empty_log_dic(log_time, market_id)
        self.current_log_dics[market_id] = current_log_dic

    def process_execution_log(self, log: ExecutionLog) -> None:
        """process execution log.

        add execution log to execution_dics.

        Args:
            log (ExecutionLog): execution log.
        """
        market_id: Market = log.market_id
        execution_price: Optional[float] = log.price
        execution_volume: int = log.volume
        if market_id not in self.execution_dics.keys():
            self.execution_dics[market_id] = {
                self._convert_price2str(execution_price): execution_volume
            }
        else:
            if execution_price in self.execution_dics[market_id].keys():
                self.execution_dics[market_id][
                    self._convert_price2str(execution_price)
                ] += execution_volume
            else:
                self.execution_dics[market_id][
                    self._convert_price2str(execution_price)
                ] = execution_volume

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        market: Market = log.market
        market_id: Market = market.market_id
        current_log_dic: dict[str, dict[str, list | dict | str]] = \
            self.current_log_dics[market_id]
        self._write_prices(market, current_log_dic)
        self._write_executions(
            self.execution_dics[market_id],
            current_log_dic
        )
        current_buy_order_book: OrderBook = market.buy_order_book
        current_buy_price_volume_dic: dict[Optional[float], int] = \
            current_buy_order_book.get_price_volume()
        self._write_order_book_diffs(
            current_log_dic,
            current_price_volume_dic=current_buy_price_volume_dic,
            previous_price_volume_dic=self.previous_buy_price_volume_dic,
            is_buy=True
        )
        self._write_order_book(
            current_log_dic, current_buy_price_volume_dic, is_buy=True
        )
        current_sell_order_book: OrderBook = market.sell_order_book
        current_sell_price_volume_dic: dict[Optional[float], int] = \
            current_sell_order_book.get_price_volume()
        self._write_order_book(
            current_log_dic, current_sell_price_volume_dic, is_buy=False
        )
        self.logs_dic[market_id].append(
            self._convert_dic2str(current_log_dic)
        )

    def _create_empty_log_dic(
        self,
        log_time: int,
        market_id: int
    ) -> dict[str, dict[str, list | dict | str]]:
        empty_log_dic: dict[str, list | dict | str] = {
            "Data": {
                "time": f"{str(log_time)}",
                "code": f"{str(market_id)}",
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

    def _add_price_volume_dic(
        self,
        market_id: MarketID,
        buy_order_book: OrderBook,
        sell_order_book: OrderBook
    ) -> None:
        self.previous_buy_price_volume_dic[market_id] = \
            buy_order_book.get_price_volume()
        self.previous_sell_price_volume_dic[market_id] = \
            sell_order_book.get_price_volume()

    def _convert_dic2str(
        self,
        dic: dict
    ) -> str:
        dic_str: str = str(dic)
        dic_str = dic_str.replace(" ", "")
        return dic_str

    def _convert_price2str(self, price: Optional[float]) -> str:
        if price is None:
            price_str: str = "NaN"
        else:
            price_str: str = f"{price:.{self.significant_figures}f}"
        return price_str

    def _write_prices(
        self,
        market: Market,
        log_dic: dict[str, list | dict | str]
    ) -> None:
        market_price: Optional[float | str] = market.get_last_executed_price()
        log_dic["Data"]["market_price"] = \
            f"{self._convert_price2str(market_price)}"
        best_buy_price: Optional[float | str] = market.get_best_buy_price()
        log_dic["Data"]["best_bid"] = \
            f"{self._convert_price2str(best_buy_price)}"
        best_sell_price: Optional[float | str] = market.get_best_sell_price()
        log_dic["Data"]["best_ask"] = \
            f"{self._convert_price2str(best_sell_price)}"
        mid_price: Optional[float | str] = market.get_mid_price()
        log_dic["Data"]["mid_price"] = \
            f"{self._convert_price2str(mid_price)}"
        return mid_price

    def _write_executions(
        self,
        execution_dic: dict[str, int],
        log_dic: dict[str, dict[str, list | dict | str]]
    ) -> None:
        for price_str, volume in execution_dic.items():
            log_dic["Data"]["message"].append(
                {"tag":"1P", "price":str(price_str)}
            )
            log_dic["Data"]["message"].append(
                {"tag":"VL", "volume":str(volume)}
            )

    def _write_order_book_diffs(
        self,
        log_dic: dict[str, list | dict | str],
        current_price_volume_dic: dict[Optional[float], int],
        previous_price_volume_dic: dict[Optional[float], int],
        is_buy: bool
    ) -> None:
        pass

    def _write_order_book(
        self,
        log_dic: dict[str, list | dict | str],
        volume_price_dic: dict[Optional[float], int],
        is_buy: bool
    ) -> None:
        str_volume_price_dic: dict[str, str] = {}
        for price, volume in volume_price_dic.items():
            if price is None:
                price_str: str = "MO"
            else:
                price_str: str = self._convert_price2str(price)
            str_volume_price_dic[f"{price_str}"] = f"{volume}"
        if is_buy:
            log_dic["buy_book"] = str_volume_price_dic
        else:
            log_dic["sell_book"] = str_volume_price_dic
