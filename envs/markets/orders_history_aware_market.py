from pams.logs import Logger
from pams.market import Market
from pams.order import Order
from pams.simulator import Simulator
import pandas as pd
from pandas import DataFrame
import random
from typing import Optional


class OrdersHistoryAwareMarket(Market):
    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            market_id, prng, simulator, name, logger=logger
        )
        self.order_history_df = pd.DataFrame(
            columns=[
                "order_price",
                "volume",
                "best_ask_price",
                "best_bid_price",
                "best_ask_volume",
                "best_bid_volume",
            ]
        )

    def _add_order(self, order):
        result = super()._add_order(order)
        self._update_order_history(order)
        return result

    def _update_order_history(self, order: Order):
        best_ask_price: float = self.get_best_sell_price()
        best_bid_price: float = self.get_best_buy_price()
        sell_order_book: dict[Optional[float], int] = self.get_sell_order_book()
        best_ask_volume: int = sell_order_book.get(best_ask_price, 0)
        buy_order_book: dict[Optional[float], int] = self.get_buy_order_book()
        best_bid_volume: int = buy_order_book.get(best_bid_price, 0)
        self.order_history_df = pd.concat(
            [
                self.order_history_df,
                pd.DataFrame(
                    {
                        "order_price": [order.price],
                        "volume": [order.volume],
                        "best_ask_price": [best_ask_price],
                        "best_bid_price": [best_bid_price],
                        "best_ask_volume": [best_ask_volume],
                        "best_bid_volume": [best_bid_volume],
                    }
                ),
            ],
            ignore_index=True,
        )