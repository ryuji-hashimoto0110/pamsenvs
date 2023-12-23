import math
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import Order
from pams.simulator import Simulator
from pams.utils import JsonRandom
from typing import Any, Optional
from typing import TypeVar
import random

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class aFCNAgent(Agent):
    """asymmetric FCN Agent (aFCNAgent) class
    """
    def __init__(
        self,
        agent_id: AgentID,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger]
    ) -> None:
        super().__init__(agent_id, prng, simulator, name, logger)

    def is_finite(self, x: float) -> bool:
        return not math.isnan(x) and not math.isinf(x)

    def setup(
        self,
        settings: dict[str, Any],
        accessible_market_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """agent setup. Usually be called from simulator / runner automatically.

        Args:
            settings (dict[str, Any]): agent configuration. Thie must include the parameters:
                - fundamentalWeight
                - chartWeight
                - feedbackAsymmetry
                - noiseWeight
                - noiseAsymmetry
                - noiseScale
                - timeWindowSize
                - orderMargin
                and can include
                - meanReversionTime
            accessible_market_ids (list[MarketID]): _description_
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_market_ids
        )
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.w_f: float = json_random.random(json_value=settings["fundamentalWeight"])
        self.w_c: float = json_random.random(json_value=settings["chartWeight"])
        self.a_feedback: float = json_random.random(
            json_value=settings["feedbackAsymmetry"]
        )
        self.w_n: float = json_random.random(json_value=settings["noiseWeight"])
        self.a_noise: float = json_random.random(
            json_value=settings["noiseAsymmetry"]
        )
        self.noise_scale: float = json_random.random(json_value=settings["noiseScale"])
        self.time_window_size = int(
            json_random.random(json_value=settings["timeWindowSize"])
        )
        self.order_margin: float = json_random.random(json_value=settings["orderMargin"])
        if "meanReversionTime" in settings:
            self.mean_reversion_time: int = int(
                json_random.random(json_value=settings["meanReversionTime"])
            )
        else:
            self.mean_reversion_time: int = self.time_window_size

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order | Cancel]:
        orders: list[Order | Cancel] = sum(
            [
                self.submit_orders_by_market(market=market) for market in markets
            ], []
        )
        return orders

    def submit_orders_by_market(self, market: Market) -> list[Order | Cancel]:
        pass

    def _calc_weights(self, market: Market):
        pass

    def _calc_expected_future_price(
        self,
        market: Market,
        fundamental_weight: float,
        chart_weight: float,
        noise_weight: float
    ) -> float:
        pass
