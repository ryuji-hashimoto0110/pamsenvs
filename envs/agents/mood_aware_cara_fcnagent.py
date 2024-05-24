from random import Random
from pams.logs import Logger
from pams.market import Market
from pams.simulator import Simulator
from pams.utils import json_random
from .cara_fcn_agent import CARAFCNAgent
from typing import Any, Literal
from typing import TypeVar

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class MoodAwareCARAFCNAgent(CARAFCNAgent):
    """Mood-Aware CARAFCNAgent.

    MoodAwareCARAFCNAgent requires all markets to be MoodAwareMarket.
    In other words, market accessed by this agent must have method: ".get_market_mood()"
    """
    def __init__(
        self,
        agent_id: AgentID,
        prng: Random,
        simulator: Simulator,
        name: str,
        logger: Logger | None
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            prng=prng,
            simulator=simulator,
            name=name,
            logger=logger
        )
        self.mood: Literal[0,1] = prng.choice([0,1])

    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.w_m: float = json_random.random(json_value=settings["moodWeight"])
        if "moodSensitivity" not in settings:
            raise ValueError(
                "MoodAwareCARAFCNAgent requires 'moodSensitivity' in settings"
            )
        self.mood_sensitivity: float = settings["moodSensitivity"]

    def get_agent_mood(self) -> Literal[0,1]:
        return self.mood
    
    def _calc_temporal_weights(
        self,
        market: Market,
        time_window_size: int
    ) -> list[float]:
        pass

    def _calc_expected_future_price(
        self,
        market: Market,
        weights: list[float],
        time_window_size: int
    ) -> float:
        pass

    def _change_mood(self, market: Market) -> None:
        pass
