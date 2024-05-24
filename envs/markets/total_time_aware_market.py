from pams import Market
from pams.logs import Logger
import random
from typing import Optional

class TotalTimeAwareMarket(Market):
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
        
