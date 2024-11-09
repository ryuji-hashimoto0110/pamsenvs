from typing import Any, Dict
from pams import Market
from pams.logs.base import OrderLog
from pams.order import Order

class RangeRegulatedMarket(Market):
    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:
        super().setup(settings, *args, **kwargs)
        if "regulationRange" in settings:
            self.range: float = settings["regulationRange"]
        else:
            self.range: float = 1.0

    def _add_order(self, order: Order) -> OrderLog:
        order: Order = self._regulate_order(order)
        return super()._add_order(order)
    
    def _regulate_order(self, order: Order) -> Order:
        maket_price: float = self.get_market_price()
        order_price: float = order.price
        if order.is_buy:
            min_buy_price: float = maket_price * (1 - self.range)
            if order_price is not None:
                if order_price < min_buy_price:
                    order.price = min_buy_price
        else:
            max_sell_price: float = maket_price * (1 + self.range)
            if order_price is not None:
                if max_sell_price < order_price:
                    order.price = max_sell_price
        return order
        


