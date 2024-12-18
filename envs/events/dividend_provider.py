from ..markets import LeaderAwareMarket
from pams.events import EventABC
from pams.events import EventHook
from pams.market import Market
from pams.session import Session
from pams.simulator import Simulator
from typing import TypeVar

MarketID = TypeVar("MarketID")

class DividendProvider(EventABC):
    def hook_registration(self) -> list[EventHook]:
        t: int = self.session.session_start_time + self.session.iteration_steps - 1
        event_hook = EventHook(
            event=self,
            hook_type="session",
            time=[t],
            is_before=False
        )
        return [event_hook]
    
    def hooked_after_session(
        self,
        simulator: Simulator,
        session: Session
    ) -> None:
        """initialize LeaderAwareMarket.
        """
        for market in simulator.markets:
            if isinstance(market, LeaderAwareMarket):
                print(f"{market.get_time()} DividendProvider: provide devidend.")
                for agent in simulator.agents:
                    market.provide_dividend(agent)

class DividendProviderwEverySteps(EventABC):
    def setup(self, settings, *args, **kwargs):
        if "dividendRate" not in settings:
            raise ValueError("dividendRate is required for DividendProviderwEverySteps.")
        else:
            self.dividend_rate: float = settings["dividendRate"]
        if "dividendInterval" in settings:
            self.divident_interval: int = settings["dividendInterval"]
        else:
            self.divident_interval: int = 1
        return super().setup(settings, *args, **kwargs)

    def hook_registration(self) -> list[EventHook]:
        start_time: int = self.session.session_start_time
        end_time: int = self.session.session_start_time + self.session.iteration_steps
        event_hook = EventHook(
            event=self,
            hook_type="market",
            time=[t for t in range(start_time, end_time, self.divident_interval)],
            is_before=False
        )
        return [event_hook]
    
    def hooked_after_step_for_market(
        self,
        simulator: Simulator,
        market: Market
    ) -> None:
        p_f: float = market.get_fundamental_price()
        dividend: float = self.dividend_rate * p_f
        for agent in simulator.agents:
            market_id: MarketID = market.market_id
            asset_volume: int = agent.get_asset_volume(market_id)
            agent.cash_amount += dividend * asset_volume

        