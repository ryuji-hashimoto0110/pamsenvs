from ..markets import YesterdayAwareMarket
from pams.market import Market
from pams.logs import Logger
from pams.order import Order
from pams.order_book import OrderBook
from pams.runners.sequential import SequentialRunner
from pams.simulator import Simulator
import random
from typing import Optional
import warnings

class SimulatorGivenRunner(SequentialRunner):
    """SimulatorGivenRunner class.
    
    SimulatorGivenRunner inherits a given simulator and enables more flexible
    initialization of a simulation.
    """
    def _setup(
        self,
        previous_simulator: Optional[Simulator] = None,
        new_logger: Optional[Logger] = None
    ) -> None:
        if previous_simulator is not None:
            self.simulator = previous_simulator
            self._initialize_times()
            self._re_setup_agents()
        else:
            super()._setup()
            self._pending_setups = []
        if new_logger is not None:
            self._assign_new_logger_to_all_classes(new_logger)

    def set_seed(self, seed: int) -> None:
        self._prng = random.Random(seed)
        self.simulator._prng = random.Random(seed)
        for agent in self.simulator.agents:
            agent.prng = random.Random(seed)
        for market in self.simulator.markets:
            market._prng = random.Random(seed)
        for session in self.simulator.sessions:
            session.prng = random.Random(seed)

    def _re_setup_agents(self) -> None:
        for agent in self.simulator.agents:
            if not hasattr(agent, "settings"):
                raise AttributeError("Please remain 'settings' in agents.")
            if not hasattr(agent, "accessible_markets_ids"):
                raise AttributeError(
                    "Please remain 'accessible_markets_ids' in agents."
                )
            for market_id in agent.accessible_markets_ids:
                del agent.asset_volumes[market_id]
            agent.setup(
                settings=agent.settings,
                accessible_markets_ids=agent.accessible_markets_ids,
            )

    def _assign_new_logger_to_all_classes(self, new_logger):
        self.logger = new_logger
        for agent in self.simulator.agents:
            agent.logger = new_logger
        for market in self.simulator.markets:
            market.logger = new_logger
        for session in self.simulator.sessions:
            session.logger = new_logger

    def _initialize_times(self):
        for market in self.simulator.markets:
            reversing_time: int = market.time + 1
            self._step_date_on_market(
                market, reversing_time
            )

    def _step_date_on_market(
        self,
        market: Market,
        reversing_time: int
    ) -> None:
        if not isinstance(market, YesterdayAwareMarket):
            warnings.warn(
                "market is not yesterday-aware. market prices are forgotten."
            )
            market.time -= reversing_time
            self._reverse_time_on_orderbook(
                market.buy_order_book, reversing_time
            )
            self._reverse_time_on_orderbook(
                market.sell_order_book, reversing_time
            )
        else:
            market._step_date(reversing_time)

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

