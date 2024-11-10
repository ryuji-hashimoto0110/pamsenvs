from ..markets import LeaderAwareMarket
from pams.events import EventABC
from pams.events import EventHook
from pams.session import Session
from pams.simulator import Simulator

class LeaderAwareMarketInitializer(EventABC):
    def hook_registration(self) -> list[EventHook]:
        t: int = self.session.session_start_time
        event_hook = EventHook(
            event=self,
            hook_type="session",
            time=[t],
            is_before=True
        )
        return [event_hook]
    
    def hooked_before_session(
        self,
        simulator: Simulator,
        session: Session
    ) -> None:
        """initialize LeaderAwareMarket.
        """
        for market in simulator.markets:
            if isinstance(market, LeaderAwareMarket):
                print(f"{market.get_time()} LeaderAwareMarketInitializer: init session.")
                market.init_session()
    


