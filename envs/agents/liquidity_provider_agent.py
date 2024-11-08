from pams.agents import HighFrequencyAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from typing import Any
from typing import TypeVar

MarketID = TypeVar("MarketID")

class LiquidityProviderAgent(HighFrequencyAgent):
    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """agent setup.
        
        Args:
            settings (dict[str, Any]): agent configuration. This must include the parameters:
                cashAmount (float): the initial cash amount.
                assetVolume (int): the initial asset volume.
                orderVolume (int): the volume of the order.
                halfSpread (float): half of the spread.
            accessible_markets_ids (list[MarketID]): list of accessible market ids.
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        self.order_volume: int = settings["orderVolume"]
        self.half_spread: float = settings["halfSpread"]

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order | Cancel]:
        """submit orders.
        """
        orders: list[Order | Cancel] = sum(
            [
                self.submit_orders_by_market(market=market) for market in markets
            ], []
        )
        return orders
    
    def submit_orders_by_market(self, market: Market) -> list[Order | Cancel]:
        """submit orders by market.
        
        LiquidityProviderAgent submits both buy and sell orders around the fundamental price.
        """
        fundamental_price: float = market.get_fundamental_price()
        buy_order: Order = Order(
            agent_id=self.agent_id,
            market_id=market.market_id,
            is_buy=True,
            price=max(0, fundamental_price-self.half_spread),
            volume=self.order_volume,
            kind=LIMIT_ORDER,
            ttl=1
        )
        sell_order: Order = Order(
            agent_id=self.agent_id,
            market_id=market.market_id,
            is_buy=False,
            price=max(0, fundamental_price+self.half_spread),
            volume=self.order_volume,
            kind=LIMIT_ORDER,
            ttl=1
        )
        orders: list[Order | Cancel] = [buy_order, sell_order]
        return orders