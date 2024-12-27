import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve()
root_path: Path = curr_path.parents[1]
from envs.markets import FCWeightsAwareMarket
from envs.markets import MoodAwareMarket
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import ExpirationLog
from pams.logs import Logger
from pams.logs import OrderLog
from pams.logs.base import SimulationBeginLog, SimulationEndLog
from pams.market import Market
from pams.order_book import OrderBook
from pams.simulator import Simulator
from typing import Optional
from typing import TypeVar

MarketID = TypeVar("MarketID")
MarketName = TypeVar("MarketName")

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
    }
    """
    def __init__(
        self,
        significant_figures: int = 1,
        is_execution_only: bool = True,
        session1_end_time: Optional[int] = None,
        session2_start_time: Optional[int] = None,
        txt_save_folder_path: Optional[Path] = None,
        txt_file_name_dic: Optional[dict[MarketName, str]] = None
    ) -> None:
        """initialization

        There are 2 sessions in Japan's stock market in 1 day.

        Args:
            significant_figures (int): significant figures to store prices. defaut to 1.
            is_execution_only (bool):
            session1_end_time (Optional[int]): specify according to the pams config.
            session2_start_time (Optional[int])
        """
        super().__init__()
        self.logs_dic: dict[MarketID, list[str]] = {}
        self.market_dic: dict[MarketID, Market] = {}
        self.market_id2name_dic: dict[MarketID, MarketName] = {}
        self.buy_order_book_dic: dict[MarketID, OrderBook] = {}
        self.sell_order_book_dic: dict[MarketID, OrderBook] = {}
        self.significant_figures: int = significant_figures
        self.is_execution_only: bool = is_execution_only
        self.session1_end_time: Optional[int] = session1_end_time
        self.session2_start_time: Optional[int] = session2_start_time
        if (
            self.session1_end_time is not None or self.session2_start_time is not None
        ):
            if (
                self.session1_end_time is None or self.session2_start_time is None
            ):
                raise ValueError(
                    "specify both session1_end_time and session2_start_time."
                )
        self.txt_save_folder_path: Optional[Path] = txt_save_folder_path
        self.txt_file_name_dic: Optional[dict[MarketName, str]] = txt_file_name_dic
        if (
            self.txt_save_folder_path is not None or self.txt_file_name_dic is not None
        ):
            if (
                self.txt_save_folder_path is None or self.txt_file_name_dic is None
            ):
                raise ValueError(
                    "specify both txt_save_folder_path and txt_file_names."
                )

    def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
        """process simulation begin log.

        store market and buy/sell order book.

        Args:
            log (MarketStepBeginLog): market step begin log.
        """
        simulator: Simulator = log.simulator
        for market in simulator.markets:
            market_id: MarketID = market.market_id
            self.logs_dic[market_id] = []
            self.market_dic[market_id] = market
            self.buy_order_book_dic[market_id] = market.buy_order_book
            self.sell_order_book_dic[market_id] = market.sell_order_book
            self._add_market_id2name_dic(market)
            if self.txt_file_name_dic is not None:
                if market.name not in self.txt_file_name_dic.keys():
                    raise ValueError(
                        f"market:{market.name} is not in txt_file_name_dic"
                    )
            
    def _add_market_id2name_dic(self, market: Market) -> None:
        market_id: MarketID = market.market_id
        market_name: MarketName = market.name
        self.market_id2name_dic[market_id] = market_name

    def _prepare_log_dic(
        self,
        log: CancelLog | ExecutionLog | ExpirationLog | OrderLog
    ) -> dict[str, dict[str, dict[str, list | dict, str]]]:
        """prepare base log_dic.

        write all informations except for messages.

        Args:
            log (CancelLog | ExecutionLog | ExpirationLog | OrderLog)

        Returns:
            dict[str, dict[str, dict[str, list | dict, str]]]
        """
        if isinstance(log, CancelLog):
            log_time: int = log.cancel_time
        else:
            log_time: int = log.time
        market_id: int = log.market_id
        market: Market = self.market_dic[market_id]
        if log_time != market.get_time():
            raise ValueError(
                "FlexSaver needs to write log concurrently with the event occurr but found " +
                f"log_time={log_time}, market.time={market.get_time()} in {log}. " +
                "please rewrite pams/market.py as in README.md."
            )
        log_dic = self._create_empty_log_dic(log_time, market_id)
        buy_volume_price_dic: dict[Optional[float], int] = \
            self.buy_order_book_dic[market_id].get_price_volume()
        sell_volume_price_dic: dict[Optional[float], int] = \
            self.sell_order_book_dic[market_id].get_price_volume()
        self._bulk_write(
            log_dic,
            market,
            buy_volume_price_dic,
            sell_volume_price_dic
        )
        return log_dic

    def _process_log_but_execution(
        self,
        log: CancelLog | ExpirationLog | OrderLog
    ) -> dict[str, dict[str, dict[str, list | dict, str]]]:
        if self.is_execution_only:
            pass
        else:
            market_id: int = log.market_id
            log_dic: dict[str, dict[str, dict[str, list | dict, str]]] = self._prepare_log_dic(log)
            self.logs_dic[market_id].append(
                self._convert_dic2str(log_dic)
            )

    def process_order_log(self, log: OrderLog) -> None:
        self._process_log_but_execution(log)

    def process_cancel_log(self, log: CancelLog) -> None:
        self._process_log_but_execution(log)

    def process_expiration_log(self, log: ExpirationLog) -> None:
        self._process_log_but_execution(log)

    def process_execution_log(self, log: ExecutionLog) -> None:
        """process execution log.

        add execution log logs_dic.

        Args:
            log (ExecutionLog): execution log.
        """
        market_id: int = log.market_id
        log_dic: dict[str, dict[str, dict[str, list | dict, str]]] = self._prepare_log_dic(log)
        execution_price: float = log.price
        execution_price_str: str = self._convert_price2str(execution_price)
        execution_volume: int = int(log.volume)
        log_dic["Data"]["message"].append(
            {"tag":"1P", "price":str(execution_price_str)}
        )
        log_dic["Data"]["message"].append(
            {"tag":"VL", "volume":str(execution_volume)}
        )
        self.logs_dic[market_id].append(
            self._convert_dic2str(log_dic)
        )

    def _create_empty_log_dic(
        self,
        log_time: int,
        market_id: int
    ) -> dict[str, dict[str, dict[str, list | dict, str]]]:
        empty_log_dic: dict[str, dict[str, list | dict, str]] = {
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
        if self.session1_end_time is not None:
            if log_time <= self.session1_end_time:
                empty_log_dic["Data"]["session_id"] = "1"
            elif self.session2_start_time <= log_time:
                empty_log_dic["Data"]["session_id"] = "2"
            else:
                raise ValueError(f"cannot identify session. time={log_time}")
        return empty_log_dic

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
        log_dic: dict[str, dict[str, list | dict, str]],
        market: Market
    ) -> None:
        if isinstance(market, MoodAwareMarket):
            mood: float = market.get_market_mood()
            mood_str: str = f"{mood:.3f}"
            log_dic["Data"]["mood"] = mood_str
        if isinstance(market, FCWeightsAwareMarket):
            wc_rate: float = market.wc_rate
            wc_rate_str: str = f"{wc_rate:.5f}"
            log_dic["Data"]["wc_rate"] = wc_rate_str
            time_window_size: int = sorted(
                market.previous_time_window_sizes
            )[len(market.previous_time_window_sizes) // 2]
            log_dic["Data"]["time_window_size"] = str(time_window_size)
        market_price: Optional[float | str] = market.get_last_executed_price()
        log_dic["Data"]["market_price"] = self._convert_price2str(market_price)
        best_buy_price: Optional[float | str] = market.get_best_buy_price()
        log_dic["Data"]["best_bid"] = self._convert_price2str(best_buy_price)
        best_sell_price: Optional[float | str] = market.get_best_sell_price()
        log_dic["Data"]["best_ask"] = self._convert_price2str(best_sell_price)
        mid_price: Optional[float | str] = market.get_mid_price()
        log_dic["Data"]["mid_price"] = self._convert_price2str(mid_price)
        return mid_price

    def _write_order_book(
        self,
        log_dic: dict[str, dict[str, list | dict, str]],
        volume_price_dic: dict[Optional[float], int],
        is_buy: bool
    ) -> None:
        str_volume_price_dic: dict[str, str] = {}
        for price, volume in volume_price_dic.items():
            if price is None:
                price_str: str = "MO"
            else:
                price_str: str = self._convert_price2str(price)
            str_volume_price_dic[f"{price_str}"] = f"{int(volume)}"
        if is_buy:
            log_dic["Data"]["buy_book"] = str_volume_price_dic
        else:
            log_dic["Data"]["sell_book"] = str_volume_price_dic

    def _bulk_write(
        self,
        log_dic: dict[str, dict[str, list | dict, str]],
        market: Market,
        buy_volume_price_dic: dict[Optional[float], int],
        sell_volume_price_dic: dict[Optional[float], int]
    ) -> None:
        self._write_prices(log_dic, market)
        self._write_order_book(
            log_dic, buy_volume_price_dic, is_buy=True
        )
        self._write_order_book(
            log_dic, sell_volume_price_dic, is_buy=False
        )

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        if self.txt_save_folder_path is None:
            return None
        else:
            if not self.txt_save_folder_path.exists():
                self.txt_save_folder_path.mkdir(parents=True)
        for market_id in self.logs_dic.keys():
            logs: list[str] = self.logs_dic[market_id]
            logs_str: str = "\n".join(logs)
            market_name: MarketName = self.market_id2name_dic[market_id]
            txt_file_name: str = self.txt_file_name_dic[market_name]
            txt_save_path: Path = self.txt_save_folder_path / txt_file_name
            with open(txt_save_path, mode="w") as f:
                f.write(logs_str)
