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
                - orderVolume
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
        if "orderVolume" in settings:
            self.order_volume: int = int(
                json_random.random(json_value=settings["orderVolume"])
            )
        else:
            self.order_volume: int = 1

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
        if not self.is_market_accessible(market_id=market.market_id):
            return []
        time: int = market.get_time()
        time_window_size: int = min(time, self.time_window_size)
        weights: list[float] = self._calc_weights(market, time_window_size)
        fundamental_weight: float = weights[0]
        chart_weight: float = weights[1]
        noise_weight: float = weights[2]
        expected_future_price: float = self._calc_expected_future_price(
            market, fundamental_weight, chart_weight, noise_weight, time_window_size
        )
        assert self.is_finite(expected_future_price)
        orders: list[Order | Cancel] = self._create_order(market, expected_future_price)
        return orders

    def _calc_weights(
        self,
        market: Market,
        time_window_size: int
    ) -> list[float]:
        time: int = market.get_time()
        market_price: float = market.get_market_price()
        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_log_return: float = chart_scale * math.log(
            market_price / market.get_market_price(time - time_window_size)
        )
        chart_weight: float = self.w_c \
            + self.a_feedback / max(1e-06, 1 + chart_log_return)
        chart_weight = max(0, chart_weight)
        noise_weight: float = self.w_n \
            + self.a_noise / max(1e-06, 1 + chart_log_return)
        noise_weight = max(0, noise_weight)
        weights: list[float] = [self.w_f, chart_weight, noise_weight]
        return weights

    def _calc_expected_future_price(
        self,
        market: Market,
        fundamental_weight: float,
        chart_weight: float,
        noise_weight: float,
        time_window_size: int
    ) -> float:
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
        expected_log_return: float = (
            1.0 / (fundamental_weight + chart_weight + noise_weight)
        ) * (
            fundamental_weight * fundamental_log_return
            + chart_weight * chart_log_return
            + noise_weight * noise_log_return
        )
        assert self.is_finite(expected_log_return)
        expected_future_price: float = market_price * math.exp(
            expected_log_return * self.time_window_size
        )
        return expected_future_price

    def _create_order(
        self,
        market: Market,
        expected_fugure_price: float
    ) -> list[Order | Cancel]:
        orders: list[Order | Cancel] = []
        market_price: float = market.get_market_price()
        if market_price < expected_fugure_price:
            order_price: float = expected_fugure_price * (1 - self.order_margin)
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=self.order_volume,
                    price=order_price,
                    ttl=self.time_window_size
                )
            )
        elif expected_fugure_price < market_price:
            order_price: float = expected_fugure_price * (1 + self.order_margin)
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=self.order_volume,
                    price=order_price,
                    ttl=self.time_window_size
                )
            )
        return orders
