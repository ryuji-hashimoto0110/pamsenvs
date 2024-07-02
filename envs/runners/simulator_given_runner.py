from pams.market import Market
from pams.order import Order
from pams.order_book import OrderBook
from pams.runners.sequential import SequentialRunner
from pams.simulator import Simulator
from typing import Optional

class SimulatorGivenRunner(SequentialRunner):
    """SimulatorGivenRunner class.
    
    SimulatorGivenRunner inherits a given simulator and enables more flexible
    initialization of a simulation.
    """
    def _setup(
        self,
        previous_simulator: Optional[Simulator] = None
    ) -> None:
        if previous_simulator is None:
            super()._setup()
        else:
            self.simulator = previous_simulator
            self._assign_new_logger_to_all_classes()
            self._initialize_times()

    def _assign_new_logger_to_all_classes(self):
        for agent in self.simulator.agents:
            agent.logger = self.logger
        for market in self.simulator.markets:
            market.logger = self.logger
        for session in self.simulator.sessions:
            session.logger = self.logger

    def _initialize_times(self):
        for market in self.simulator.markets:
            reversing_time: int = market.time + 1
            self._reverse_time_on_market(
                market, reversing_time
            )

    def _reverse_time_on_market(
        self,
        market: Market,
        reversing_time: int
    ) -> None:
        market.time -= reversing_time
        self._reverse_time_on_orderbook(
            market.buy_order_book, reversing_time
        )
        self._reverse_time_on_orderbook(
            market.sell_order_book, reversing_time
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



