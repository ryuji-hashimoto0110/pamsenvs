import math
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
        self.mood: Literal[0,1] = self.prng.choice([0,1])

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
        weights: list[float] = [self.w_f, self.w_c, self.w_n, self.w_m]
        return weights

    def _calc_expected_future_price(
        self,
        market: Market,
        weights: list[float],
        time_window_size: int
    ) -> float:
        self._change_mood(market)
        fundamental_weight: float = weights[0]
        chart_weight: float = weights[1]
        noise_weight: float = weights[2]
        mood_weight: float = weights[3]
        time: int = market.get_time()
        market_price: float = market.get_market_price()
        fundamental_price: float = market.get_fundamental_price()
        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return: float = fundamental_scale * math.log(
            fundamental_price / market_price
        )
        assert self.is_finite(fundamental_log_return)
        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_log_return: float = chart_scale * math.log(
            market_price / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_log_return)
        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)
        agent_mood: Literal[0,1] = self.get_agent_mood()
        mood_log_return: float = (
            (agent_mood == 1) - (agent_mood == 0)
        )
        expected_log_return: float = (
            1.0 / (fundamental_weight + chart_weight + noise_weight + mood_weight)
        ) * (
            fundamental_weight * fundamental_log_return
            + chart_weight * chart_log_return * (1 if self.is_chart_following else -1)
            + noise_weight * noise_log_return
            + mood_weight * mood_log_return
        )
        assert self.is_finite(expected_log_return)
        expected_future_price: float = market_price * math.exp(
            expected_log_return * self.time_window_size
        )
        return expected_future_price

    def _change_mood(self, market: Market) -> None:
        market_mood: float = market.get_market_mood()
        agent_mood: Literal[0,1] = self.get_agent_mood()
        if agent_mood == 0:
            change_prob: float = self.mood_sensitivity * market_mood
            self.mood = self.prng.choice(
                [0,1],
                p=[1-change_prob, change_prob]
            )
        elif agent_mood == 1:
            change_prob: float = self.mood_sensitivity * (1 - market_mood)
            self.mood = self.prng.choce(
                [0,1],
                [change_prob, 1-change_prob]
            )
        else:
            raise ValueError(
                f"agent_mood must be either 0 or 1, but found {agent_mood}."
            )
