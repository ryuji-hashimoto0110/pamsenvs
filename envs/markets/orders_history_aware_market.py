from pams.logs import Logger
from pams.logs import Log
from pams.market import Market
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from pams.simulator import Simulator
import pandas as pd
from pandas import DataFrame
import random
from typing import Any
from typing import Optional
from typing import TypeVar

AgentID = TypeVar("AgentID")


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
                "mid_price"
            ]
        )

    def setup(
        self,
        settings: dict[str, Any],
        *args, **kwargs
    ) -> None:
        super().setup(settings, *args, **kwargs)
        if "orderHistoryDfPath" in settings:
            if "dummyAgentID" not in settings:
                raise ValueError(
                    "Specify 'dummyAgentID' to pre-add orders from the history DataFrame."
                )
            self.dummy_agent_id: AgentID = settings["dummyAgentID"]
            df: DataFrame = pd.read_csv(
                settings["orderHistoryDfPath"], index_col=0
            )
            self.time = 0
            self._mid_prices.append(None)
            self._n_buy_orders.append(0)
            self._n_sell_orders.append(0)
            self._pre_add_orders(df)
            self.time = -1

    def _pre_add_orders(self, df: DataFrame) -> None:
        """add orders in advence before the simulation starts.
        
        Assume the DataFrame `df` contains the following columns:
            - 'price': The price of the order.
            - 'size': The volume of the order.
            - 'side': The side of the order (1 if buy/ 0 if sell).
            - 'type': The type of the order (1 if LIMIT_ORDER/ 0 if MARKET_ORDER).
            - 'best_ask_price': The best ask price in the market.
            - 'best_bid_price': The best bid price in the market.
            - 'best_ask_volume': The volume at the best ask price.
            - 'best_bid_volume': The volume at the best bid price.

        This method 
            1) add buy / sell orders to form the initial best bid / ask prices and volumes
            2) sequentially add orders to the order book to form the initial order book.

        WARNINGS: This method triggers the self._execution() method.
        """
        self._add_initial_orders(df, is_buy=True)
        self._add_initial_orders(df, is_buy=False)
        for _, row in df.iterrows():
            order_price: float = row["price"]
            order_volume: int = row["size"]
            is_buy = (row["side"] == 1)
            order_type = LIMIT_ORDER if row["type"] == 1 else MARKET_ORDER
            order = Order(
                market_id=self.market_id,
                agent_id=self.dummy_agent_id,
                price=order_price,
                volume=order_volume,
                is_buy=is_buy,
                kind=order_type,
            )
            self._add_order(order)
            logs: list[Log] = self._execution()
            if 0 < len(logs):
                raise ValueError(
                    "logs remain after adding initial orders. " + 
                    "Maybe you forgot to rewrite pams/market.py."
                )

    def _add_initial_orders(self, df: DataFrame, is_buy: bool) -> None:
        """Add initial orders to the market based on the DataFrame `df`.
        
        This method adds orders to set the initial best bid/ask prices and volumes.
        """
        if is_buy:
            order_price: float = df["best_bid_price"].iloc[0]
            order_volume: int = df["best_bid_volume"].iloc[0]
        else:
            order_price = df["best_ask_price"].iloc[0]
            order_volume = df["best_ask_volume"].iloc[0]
        order = Order(
            market_id=self.market_id,
            agent_id=self.dummy_agent_id,
            price=order_price,
            volume=order_volume,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
        )
        self._add_order(order)

    def _add_order(self, order):
        result = super()._add_order(order)
        self._update_order_history(order)
        return result

    def _update_order_history(self, order: Order):
        best_ask_price: Optional[float] = self.get_best_sell_price()
        best_bid_price: Optional[float] = self.get_best_buy_price()
        mid_price: Optional[float] = self.get_mid_price()
        mid_price = self.get_market_price() if mid_price is None else mid_price
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
                        "mid_price": [mid_price],
                    }
                ),
            ],
            ignore_index=True,
        )