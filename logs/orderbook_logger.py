import cv2
import io
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from numpy import ndarray
from pams.logs import Logger, SimulationEndLog
from pams.logs import ExecutionLog, MarketStepBeginLog, MarketStepEndLog, OrderLog
from pams.market import Market
from pams.order_book import OrderBook
import pandas as pd
from pandas import DataFrame, Series
import pathlib
from pathlib import Path
from typing import Optional, TypeVar

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class OrderBookSaver(Logger):
    """saver of the market order books logger.

    make video of snapshots of order books and price changes.
    """
    def __init__(
        self,
        videos_path: Path,
        specific_agent_color_dic: dict[AgentID, str] = {},
        draw_tick_num: int = 10,
        fps: int = 10,
        video_width: int = 256,
        video_height: int = 256,
        dupulicate_frames_num: int = 5
    ) -> None:
        super().__init__()
        self.videos_path: Path = videos_path
        if not self.videos_path.exists():
            self.videos_path.mkdir(parents=True)
        self.specific_agent_color_dic: dict[AgentID, str] = specific_agent_color_dic
        self.draw_tick_num: int = draw_tick_num
        self.fps: int = fps
        self.video_width: int = video_width
        self.video_height: int = video_height
        self.dupulicate_frames_num = dupulicate_frames_num
        self.video_writer_dic: dict[MarketID, cv2.VideoWriter] = {}
        self.buy_order_book_dic: dict[MarketID, OrderBook] = {}
        self.sell_order_book_dic: dict[MarketID, OrderBook] = {}
        self.buy_price_volume_dic: dict[MarketID, dict[Optional[float], int]] = {}
        self.sell_price_volume_dic: dict[MarketID, dict[Optional[float], int]] = {}
        self.market_price_dic: dict[str | MarketID, list[int | float]] = {}
        self.ticksize_dic: dict[MarketID, float] = {}

    def _convert_orderbook2df(
        self,
        buy_price_volume_dic: dict[Optional[float], int],
        sell_price_volume_dic: dict[Optional[float], int],
        tick_size: float
    ) -> Optional[DataFrame]:
        """convert order books t DataFrame

        The result of this method is orderbook_df that looks like as follows.
        For more detail description, see explain_drawing_orderbook.ipynb.

        If there are no orders in order books at all, return None.

        | index | limit order volume | market order volume |
        |-------|--------------------|---------------------|
        |  OVER |      -1000         |         ...         |
        |   ... |        ...         |         ...         |
        |  1001 |       -100         |         100         |
        |  1000 |          0         |           0         |
        |   999 |        200         |        -100         |
        |   ... |        ...         |         ...         |
        | UNDER |       1500         |         ...         |

        Args:
            buy_order_book (OrderBook): _description_
            sell_order_book (OrderBook): _description_

        Returns:
            orderbook_df: series that describes the order books.
        """
        best_buy_price, mid_price, best_sell_price = self._get_representative_prices(
            buy_price_volume_dic, sell_price_volume_dic, tick_size
        )
        if mid_price == None:
            return None
        orderbook_df: DataFrame = self._initialize_orderbook_df(mid_price, tick_size)
        orderbook_df = self._add_marketorders2orderbook_df(
            orderbook_df, buy_price_volume_dic, best_sell_price, is_buy=True
        )
        orderbook_df = self._add_marketorders2orderbook_df(
            orderbook_df, sell_price_volume_dic, best_buy_price, is_buy=False
        )
        orderbook_df = self._add_limitorders2orderbook_df(
            orderbook_df, buy_price_volume_dic, sell_price_volume_dic
        )
        return orderbook_df

    def _extract_limit_prices(
        self,
        price_volume_dic: dict[Optional[float], int]
    ) -> list[float]:
        """extract limit prices without "None" from order book dictionary.

        Args:
            price_volume_dic (dict[Optional[float], int]): _description_

        Returns:
            prices (list[float]): _description_
        """
        prices: list[Optional[float]] = list(price_volume_dic.keys())
        prices: list[float] = [price for price in prices if price != None]
        return prices

    def _get_best_price(self, prices: list[float], is_buy: bool) -> Optional[float]:
        """get best quote price from limit price list.

        Args:
            prices (list[float]): _description_
            is_buy (bool): _description_

        Returns:
            Optional[float]: _description_
        """
        best_price: Optional[float] = None
        if 0 < len(prices):
            best_price = max(prices) if is_buy else min(prices)
        return best_price

    def _modify_price(self, price: float, tick_size: float) -> float:
        """modify mid price according to tick size

        Args:
            price (float): _description_
            tick_size (float): _description_

        Returns:
            mid_price (float): _description_
        """
        return tick_size * math.ceil(price / tick_size)

    def _round_to_significant_digit(
        self,
        num: float,
        tick_size: float
    ) -> float:
        tick_str: str = str(tick_size)
        if "." in tick_str:
            decimal_places: int = len(tick_str.split('.')[1])
        else:
            decimal_places = 0
        return round(num, decimal_places)

    def _get_representative_prices(
        self,
        buy_price_volume_dic: dict[Optional[float], int],
        sell_price_volume_dic: dict[Optional[float], int],
        tick_size: float
    ) -> tuple[Optional[float]]:
        """extract best quote price and mid price.

        Args:
            buy_price_volume_dic (dict[Optional[float], int]): _description_
            sell_price_volume_dic (dict[Optional[float], int]): _description_
            tick_size (float): _description_

        Returns:
            tuple[Optional[float]]: _description_
        """
        buy_prices: list[float] = self._extract_limit_prices(buy_price_volume_dic)
        sell_prices: list[float] = self._extract_limit_prices(sell_price_volume_dic)
        best_buy_price: Optional[float] = self._get_best_price(buy_prices, is_buy=True)
        best_sell_price: Optional[float] = self._get_best_price(sell_prices, is_buy=False)
        mid_price: Optional[float] = None
        if (best_sell_price == None) and (best_buy_price != None):
            mid_price = best_buy_price + tick_size
            best_sell_price = mid_price
        elif (best_sell_price != None) and (best_buy_price == None):
            mid_price = best_sell_price - tick_size
            best_buy_price = mid_price
        elif (best_sell_price != None) and (best_buy_price != None):
            mid_price = (best_sell_price + best_buy_price) / 2
        if mid_price != None:
            mid_price: float = self._modify_price(mid_price, tick_size)
            mid_price = self._round_to_significant_digit(mid_price, tick_size)
            best_buy_price = self._round_to_significant_digit(best_buy_price, tick_size)
            best_sell_price = self._round_to_significant_digit(best_sell_price, tick_size)
        return best_buy_price, mid_price, best_sell_price

    def _initialize_orderbook_df(
        self,
        mid_price: float,
        tick_size: float
    ) -> DataFrame:
        prices_arr: ndarray = np.arange(
            mid_price - self.draw_tick_num * tick_size,
            mid_price + self.draw_tick_num * tick_size + tick_size,
            tick_size,
            dtype=np.float32
        )
        prices_list: list[str | float] = \
            ["OVER"] + list(np.sort(prices_arr)[::-1]) + ["UNDER"]
        prices_list = [
            self._round_to_significant_digit(p, tick_size) if not isinstance(p, str) else p for p in prices_list
        ]
        orderbook_df: DataFrame = pd.DataFrame(
            data=np.zeros((len(prices_list), 3)),
            index=prices_list,
            columns=["limit buy order volume", "limit sell order volume", "market order volume"]
        )
        return orderbook_df

    def _add_marketorders2orderbook_df(
        self,
        orderbook_df: DataFrame,
        price_volume_dic: dict[Optional[float], int],
        best_price: float,
        is_buy: bool
    ) -> DataFrame:
        market_order_volume: int = 0
        if None in price_volume_dic.keys():
            market_order_volume: int = price_volume_dic[None]
        if best_price in orderbook_df.index:
            orderbook_df.loc[best_price, "market order volume"] = \
                market_order_volume if is_buy else - market_order_volume
        return orderbook_df

    def _add_limitorders2orderbook_df(
        self,
        orderbook_df: DataFrame,
        buy_price_volume_dic: dict[Optional[float], int],
        sell_price_volume_dic: dict[Optional[float], int]
    ) -> DataFrame:
        for buy_price in buy_price_volume_dic.keys():
            if buy_price == None:
                continue
            if buy_price in orderbook_df.index:
                orderbook_df.loc[buy_price, "limit buy order volume"] = buy_price_volume_dic[buy_price]
            else:
                orderbook_df.loc["UNDER", "limit buy order volume"] += buy_price_volume_dic[buy_price]
        for sell_price in sell_price_volume_dic.keys():
            if sell_price == None:
                continue
            if sell_price in orderbook_df.index:
                orderbook_df.loc[sell_price, "limit sell order volume"] = - sell_price_volume_dic[sell_price]
            else:
                orderbook_df.loc["OVER", "limit sell order volume"] -= sell_price_volume_dic[sell_price]
        return orderbook_df

    def _add_ticksize(self, market: Market) -> None:
        market_id: int = market.market_id
        self.ticksize_dic[market_id] = market.tick_size

    def _add_order_books(self, market: Market) -> None:
        market_id: int = market.market_id
        self.buy_order_book_dic[market_id] = market.buy_order_book
        self.sell_order_book_dic[market_id] = market.sell_order_book

    def _add_price_volume_dic(
        self,
        market_id: MarketID,
        buy_order_book: OrderBook,
        sell_order_book: OrderBook
    ) -> None:
        self.buy_price_volume_dic[market_id] = buy_order_book.get_price_volume()
        self.sell_price_volume_dic[market_id] = sell_order_book.get_price_volume()

    def _add_videowriter(self, market: Market) -> None:
        market_name: str = market.name
        market_id: int = market.market_id
        video_name: str = f"{market_name}.mp4"
        video_path: Path = self.videos_path / video_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.fps,
            (self.video_width, self.video_height)
        )
        self.video_writer_dic[market_id] = video_writer

    def _draw_base_orderbook(
        self,
        ax: Axes,
        orderbook_df: Optional[DataFrame],
        tick_size: float
    ) -> Optional[ndarray]:
        ax.grid()
        if orderbook_df is None:
            return
        prices_w_str: ndarray = orderbook_df.index.values
        prices: ndarray = prices_w_str.copy()
        prices[0] = prices_w_str[1] + tick_size
        prices[-1] = prices_w_str[-2] - tick_size
        mid_price: float = prices[math.ceil(len(prices)/2)]
        limit_sell_volumes: Series = orderbook_df["limit sell order volume"]
        limit_buy_volumes: Series = orderbook_df["limit buy order volume"]
        marketorder_volumes: Series = orderbook_df["market order volume"]
        ax.vlines(0, prices[0], prices[-1], colors="black")
        ax.barh(prices, limit_buy_volumes, align="center", height=tick_size,
                color="gray", edgecolor="black", alpha=0.5)
        ax.barh(prices, limit_sell_volumes, align="center", height=tick_size,
                color="gray", edgecolor="black", alpha=0.5)
        ax.barh(prices, marketorder_volumes, align="center", height=tick_size,
                color="gray", edgecolor="black", alpha=0.5)
        ax.set_yticks(list(prices))
        ax.set_yticklabels(list(prices_w_str))
        max_volume: int = \
            (limit_sell_volumes.abs() + limit_buy_volumes + marketorder_volumes).sort_values().values[-1]
        ax.set_xlim([-max_volume, max_volume])
        ax.hlines(mid_price, -max_volume, max_volume, color="black")
        for p in ["left", "top", "right", "bottom"]:
            ax.spines[p].set_visible(False)
            if p == "left":
                ax.spines[p].set_position("zero")
        prices = np.array(
            [
                self._round_to_significant_digit(float(p), tick_size) for p in prices
            ]
        )
        return prices

    def _plot_prices(
        self,
        ax: Axes,
        market_id: MarketID
    ):
        ticks: list[int] = self.market_price_dic["ticks"]
        market_prices: list[float] = self.market_price_dic[market_id]
        ax.plot(ticks, market_prices)
        ax.set_xlabel("tick")
        ax.set_ylabel("market price")

    def _put_color(
        self,
        ax: Axes,
        color: str,
        prices: ndarray,
        price: float
    ) -> None:
        labels = ax.get_yticklabels()
        if price in prices:
            price_idx: int = np.where(prices == price)[0][0]
        elif price < np.min(prices):
            price_idx: int = len(prices) - 1
        elif np.max(prices) < price:
            price_idx: int = 0
        labels = ax.get_yticklabels()
        labels[price_idx].set_color(color)

    def _draw_base_fig(
        self,
        market_id: MarketID,
        orderbook_df: Optional[DataFrame],
        t: int,
        tick_size: float
    ):
        fig = plt.figure(figsize=(15,15), dpi=180, facecolor="w")
        fig.suptitle(f"t={t}", size=15)
        orderbook_ax: Axes = fig.add_subplot(1,2,1)
        orderbook_ax.set_title("ASK     BID", size=20)
        fig.subplots_adjust(top=0.92, hspace=0.2)
        prices: ndarray = self._draw_base_orderbook(orderbook_ax, orderbook_df, tick_size)
        price_ax: Axes = fig.add_subplot(3,2,4)
        price_ax.set_title("price change", size=20)
        self._plot_prices(price_ax, market_id)
        return fig, orderbook_ax, price_ax, prices

    def _save_fig(
        self,
        fig: plt.figure,
        market_id: MarketID
    ) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.resize(img, (self.video_width, self.video_height))
        for _ in range(self.dupulicate_frames_num):
            self.video_writer_dic[market_id].write(img)

    def process_order_log(self, log: OrderLog) -> None:
        market_id: MarketID = log.market_id
        tick_size: float = self.ticksize_dic[market_id]
        t: int = log.time
        buy_order_book: OrderBook = self.buy_order_book_dic[market_id]
        sell_order_book: OrderBook = self.sell_order_book_dic[market_id]
        self._add_price_volume_dic(
            market_id, buy_order_book, sell_order_book
        )
        orderbook_df: DataFrame = self._convert_orderbook2df(
            self.buy_price_volume_dic[market_id],
            self.sell_price_volume_dic[market_id],
            tick_size
        )
        fig, orderbook_ax, _, prices = self._draw_base_fig(market_id, orderbook_df, t, tick_size)
        agent_id: AgentID = log.agent_id
        if agent_id in self.specific_agent_color_dic.keys():
            is_buy: bool = log.is_buy
            price: float = log.price
            if price == None:
                price = min(self.sell_price_volume_dic[market_id].values()) if is_buy \
                    else max(self.buy_price_volume_dic[market_id].values())
            price = self._round_to_significant_digit(price, tick_size)
            color: str = self.specific_agent_color_dic[agent_id]
            self._put_color(
                orderbook_ax, color, prices, price
            )
        self._save_fig(fig, market_id)
        plt.clf()
        plt.close()

    def process_execution_log(self, log: ExecutionLog) -> None:
        market_id: MarketID = log.market_id
        tick_size: float = self.ticksize_dic[market_id]
        t: int = log.time
        buy_price_volume_dic_: dict[Optional[float], int] = self.buy_price_volume_dic[market_id]
        sell_price_volume_dic_: dict[Optional[float], int] = self.sell_price_volume_dic[market_id]
        orderbook_df: DataFrame = self._convert_orderbook2df(
            buy_price_volume_dic_,
            sell_price_volume_dic_,
            tick_size
        )
        fig, orderbook_ax, _, prices = self._draw_base_fig(market_id, orderbook_df, t, tick_size)
        execution_price: float = log.price
        execution_price = self._round_to_significant_digit(execution_price, tick_size)
        if execution_price in prices:
            execution_idx: int = np.where(prices == execution_price)[0][0]
        elif execution_price < np.min(prices):
            execution_idx: int = len(prices) - 1
        elif np.max(prices) < execution_price:
            execution_idx: int = 0
        labels = orderbook_ax.get_yticklabels()
        labels[execution_idx].set_color("tab:red")
        self._save_fig(fig, market_id)
        plt.clf()
        plt.close()
        buy_order_book: OrderBook = self.buy_order_book_dic[market_id]
        sell_order_book: OrderBook = self.sell_order_book_dic[market_id]
        self._add_price_volume_dic(
            market_id, buy_order_book, sell_order_book
        )
        orderbook_df: DataFrame = self._convert_orderbook2df(
            self.buy_price_volume_dic[market_id],
            self.sell_price_volume_dic[market_id],
            tick_size
        )
        fig, _, _, _ = self._draw_base_fig(market_id, orderbook_df, t, tick_size)
        self._save_fig(fig, market_id)
        plt.clf()
        plt.close()

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        market: Market = log.market
        market_id: Market = market.market_id
        if market_id not in self.video_writer_dic.keys():
            self._add_videowriter(market)
        if market_id not in self.buy_order_book_dic.keys():
            self._add_order_books(market)
        if market_id not in self.ticksize_dic.keys():
            self._add_ticksize(market)
        if "ticks" not in self.market_price_dic.keys():
            self.market_price_dic["ticks"] = []
        if market_id not in self.market_price_dic.keys():
            self.market_price_dic[market_id] = []
        tick_size: float = self.ticksize_dic[market_id]
        t: int = market.get_time()
        buy_order_book: OrderBook = self.buy_order_book_dic[market_id]
        sell_order_book: OrderBook = self.sell_order_book_dic[market_id]
        self._add_price_volume_dic(
            market_id, buy_order_book, sell_order_book
        )
        orderbook_df: DataFrame = self._convert_orderbook2df(
            self.buy_price_volume_dic[market_id],
            self.sell_price_volume_dic[market_id],
            tick_size
        )
        fig, _, _, _ = self._draw_base_fig(market_id, orderbook_df, t, tick_size)
        self._save_fig(fig, market_id)
        plt.clf()
        plt.close()

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        market: Market = log.market
        market_id: Market = market.market_id
        tick_size: float = self.ticksize_dic[market_id]
        t: int = market.get_time()
        buy_order_book: OrderBook = self.buy_order_book_dic[market_id]
        sell_order_book: OrderBook = self.sell_order_book_dic[market_id]
        self._add_price_volume_dic(
            market_id, buy_order_book, sell_order_book
        )
        orderbook_df: DataFrame = self._convert_orderbook2df(
            self.buy_price_volume_dic[market_id],
            self.sell_price_volume_dic[market_id],
            tick_size
        )
        market_price: float = market.get_market_price()
        if t not in self.market_price_dic["ticks"]:
            self.market_price_dic["ticks"].append(t)
        self.market_price_dic[market_id].append(market_price)
        fig, _, _, _ = self._draw_base_fig(market_id, orderbook_df, t, tick_size)
        self._save_fig(fig, market_id)
        plt.clf()
        plt.close()

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        for video_writer in self.video_writer_dic.values():
            video_writer.release()