from pams.logs import Logger
from pams.order import Order
from pams.order_book import OrderBook
from .total_time_aware_market import TotalTimeAwareMarket
import random
from typing import Optional
from typing import TypeVar

T = TypeVar("T")

class YesterdayAwareMarket(TotalTimeAwareMarket):
    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(
            market_id=market_id,
            prng=prng,
            simulator=simulator,
            name=name,
            logger=logger
        )
        self._yesterday_market_prices: list[Optional[float]] = []

    def get_market_price(
        self,
        time: Optional[int]
    ) -> float:
        if 0 <= time:
            return super().get_market_price(time)
        else:
            if abs(time) < len(self._yesterday_market_prices):
                return self._yesterday_market_prices[time]
            else:
                return self._market_prices[0]
        
    def _step_date(self, reversing_time: int) -> None:
        self.time -= reversing_time
        self._yesterday_market_prices = self._market_prices
        self._market_prices: list[Optional[float]] = []
        self._last_executed_prices: list[Optional[float]] = []
        self._mid_prices: list[Optional[float]] = []
        self._fundamental_prices: list[Optional[float]] = []
        self._executed_volumes: list[int] = []
        self._executed_total_prices: list[float] = []
        self._n_buy_orders: list[int] = []
        self._n_sell_orders: list[int] = []
        self._reverse_time_on_orderbook(
            self.buy_order_book, reversing_time
        )
        self._reverse_time_on_orderbook(
            self.sell_order_book, reversing_time
        )

    def _reverse_time_on_orderbook(
        self,
        order_book: OrderBook,
        reversing_time: int
    ) -> None:
        new_expire_time_list: dict[int, list[Order]] = {}
        for expire_time, orders in order_book.expire_time_list.items():
            new_expire_time_list[
                max(0, expire_time-reversing_time)
            ] = orders
        order_book.expire_time_list = new_expire_time_list

