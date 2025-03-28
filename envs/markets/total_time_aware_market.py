from pams import Market
from pams.logs import Logger
from pams.order_book import OrderBook
import random
from .range_regulated_market import RangeRegulatedMarket
from typing import Optional

class TotalTimeAwareMarket(RangeRegulatedMarket):
    """Total-time-aware Market class.
    """
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
        self.total_iteration_steps: Optional[int] = None

    def get_remaining_time(self) -> int:
        if self.total_iteration_steps is None:
            if len(self.simulator.sessions) == 0:
                return 1e+10
            else:
                self.total_iteration_steps = sum(
                    [session.iteration_steps for session in self.simulator.sessions]
                )
        return self.total_iteration_steps - self.get_time()
    
    def get_ofi(self) -> float:
        num_buy_orders: int = self.calc_num_orders(self.buy_order_book)
        num_sell_orders: int = self.calc_num_orders(self.sell_order_book)
        if num_buy_orders + num_sell_orders == 0:
            ofi: float = 0.0
        else:
            ofi: float = (num_buy_orders - num_sell_orders) / (num_buy_orders + num_sell_orders)
        return f"\\n[Order flow imbalance]market id: {self.market_id}, " + \
            f"order flow imbalance: {ofi}", ofi
        
    def calc_num_orders(self, order_book: OrderBook) -> int:
        return len(order_book.priority_queue)
