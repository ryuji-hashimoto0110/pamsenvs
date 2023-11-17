from pams.agents import HighFrequencyAgent
from pams.order import Cancel, Order, MARKET_ORDER
from pams.market import Market
from typing import Any

class WhaleAgent(HighFrequencyAgent):
    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if "isBuy" not in settings:
            raise ValueError("isBuy is required for WhaleAgent")
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for WhaleAgent")
        if "submitOrdersRate" not in settings:
            raise ValueError("submitOrdersRate is required for WhaleAgent")
        if "slicingNum" not in settings:
            raise ValueError("slicingNum is required for WhaleAgent")
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        self.is_buy: bool = settings["isBuy"]
        self.volume: int = int(settings["orderVolume"])
        self.submit_orders_rate: float = float(settings["submitOrdersRate"])
        self.slicing_num: int = float(settings["slicingNum"])
        self.remain_slicing_num: int = self.slicing_num
        self.slice_orders_now: list
        print(self.agent_id)

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order | Cancel]:
        if 0 < self.remain_slicing_num and self.remain_slicing_num < self.slicing_num:
            print(markets[0].get_time())
            orders = []
            for order in self.slice_orders_now:
                orders.append(
                    Order(
                        agent_id=order.agent_id,
                        market_id=order.market_id,
                        is_buy=order.is_buy,
                        kind=order.kind,
                        volume=self.volume,
                        price=None,
                        ttl=order.ttl,
                    )
                )
            self.remain_slicing_num -= 1
            return orders
        else:
            is_submit_orders: bool = self.prng.choices(
                [False, True], cum_weights=[1 - self.submit_orders_rate, 1], k=1
            )[0]
            self.remain_slicing_num = self.slicing_num
        if not is_submit_orders:
            return []
        orders: list[Order | Cancel] = [
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=self.is_buy,
                kind=MARKET_ORDER,
                volume=self.volume,
                price=None,
                ttl=1,
            ) for market in markets
        ]
        self.slice_orders_now = orders.copy()
        self.remain_slicing_num -= 1
        print(markets[0].get_time())
        return orders