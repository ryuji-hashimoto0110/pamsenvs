import bisect
from pams.logs import Logger
from pams.order import Order
from pams.order_book import OrderBook
from .total_time_aware_market import TotalTimeAwareMarket
import random
from typing import Iterable
from typing import Optional

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
        self.all_time_high: float = 0
        self.all_time_low: float = 0

    def get_market_price(
        self,
        time: Optional[int] = None
    ) -> float:
        if time is None:
            return super().get_market_price(time)
        elif 0 <= time:
            return super().get_market_price(time)
        else:
            if abs(time) < len(self._yesterday_market_prices):
                return self._yesterday_market_prices[time]
            else:
                return self._market_prices[0]
            
    def get_market_prices(
        self,
        times: Optional[Iterable[int]] = None
    ) -> list[float]:
        raw_market_prices: list[float] = super().get_market_prices(None)
        daily_high: float = max(raw_market_prices)
        daily_low: float = min(raw_market_prices)
        if daily_high > self.all_time_high:
            self.all_time_high = daily_high
        if daily_low < self.all_time_low:
            self.all_time_low = daily_low
        if times is None:
            return super().get_market_prices(times)
        times_list: list[int] = list(times)
        zero_idx: int = bisect.bisect_left(times_list, 0)
        if zero_idx == 0:
            return super().get_market_prices(times)
        else:
            left_times_list: list[int] = times_list[:zero_idx]
            if zero_idx == len(times_list):
                right_times_list: list[int] = []
            else:
                right_times_list: list[int] = times_list[zero_idx:]
            if len(self._yesterday_market_prices) < len(left_times_list):
                left_market_prices: list[float] = self._yesterday_market_prices
            else:
                left_market_prices: list[float] = [
                    self._yesterday_market_prices[idx] for idx in left_times_list
                ]
            right_market_prices: list[float] = [
                self._market_prices[idx] for idx in right_times_list
            ]
            market_prices: list[float] = left_market_prices + right_market_prices
            return market_prices 
        
    def _step_date(self, reversing_time: int) -> None:
        self._yesterday_market_prices = self._market_prices[:self.time]
        self.time -= reversing_time
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
            new_expire_time: int = max(0, expire_time-reversing_time)
            if new_expire_time not in new_expire_time_list:
                new_expire_time_list[new_expire_time] = []
            new_expire_time_list[new_expire_time].extend(orders)
            for order in orders:
                order.placed_at -= reversing_time
        order_book.expire_time_list = new_expire_time_list

