from .cara_fcn_agent import CARAFCNAgent
import math
from pams.logs import Logger
from pams.market import Market
from pams.simulator import Simulator
from pams.utils import JsonRandom
from typing import Any, Optional
from typing import TypeVar
import random

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class aFCNAgent(CARAFCNAgent):
    """asymmetric FCN Agent (aFCNAgent) class

    aFCNAgent's order decision mechanism is mostly based on Chiarella et al., 2008.
    The difference from the paper mentiond above is that aFCNAgent's chart/noise weight change through time
    by parameters "feedbackAsymmetry" and "noiseAsymmetry".

    References:
        - Chiarella, C., Iori, G., & Perello, J. (2009). The impact of heterogeneous trading rules
        on the limit order book and order flows,
        Journal of Economic Dynamics and Control, 33 (3), 525-537. https://doi.org/10.1016/j.jedc.2008.08.001

    note: The original paper consider market order, but aFCNAgent here is not allowed to submit market order
        because it unstabilize the simulation due to the specification of PAMS.
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

    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """agent setup. Usually be called from simulator / runner automatically.

        Args:
            settings (dict[str, Any]): agent configuration. Thie must include the parameters:
                - fundamentalWeight: weight given to the fundamentalist component.
                - chartWeight: weight given to the chartist component.
                - feedbackAsymmetry: feedback asymmetry.
                    Chart weight is amplified when the observed stock return is negative
                    by feedbackAsymmetry coefficient.
                - noiseWeight: weight given to the chartist component.
                - noiseAsymmetry: noise asymmetry.
                    Noise weight is amplified when the observed stock return is negative
                    by noiseAsymmetry coefficient.
                - noiseScale: the scale of noise component.
                - timeWindowSize: time horizon.
                - riskAversionTerm: reference level of risk aversion.
                    The precise relative risk aversion coefficient is calculated
                    by using fundamental/chart weights.
                and can include
                - chartFollowRate: probability that the agent is chart-follower.
            accessible_market_ids (list[MarketID]): _description_

        If feedbackAsymmetry and noiseAsymmetry are both 0, aFCNAgent is equivalent to FCNAgent.
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.a_feedback: float = json_random.random(
            json_value=settings["feedbackAsymmetry"]
        )
        self.a_noise: float = json_random.random(
            json_value=settings["noiseAsymmetry"]
        )

    def _calc_temporal_weights(
        self,
        market: Market,
        time_window_size: int
    ) -> list[float]:
        """calculate temporal FCN weights.

        Chartist component in FCNAgent can be regarded as positive feedback trader. Also, noise component is
        noise trader. feedback/noise traders are thought to be the factor to cause asymmetric volatility change
        because they tend to react more to the decline of stock price than price rising phase.
        aFCNAgent is implemented to reproduce this stylized fact
        by amplifying the chart/noise weight when market price is declining.

        Args:
            market (Market): market to order.
            time_window_size (int): time window size

        Returns:
            weights(list[float]): weights list. [fundamental weight, chartist weight, noise weight]
        """
        time: int = market.get_time()
        market_price: float = market.get_market_price()
        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_log_return: float = chart_scale * 100 * math.log(
            market_price / market.get_market_price(time - time_window_size)
        )
        chart_weight: float = max(
            0, self.w_c - min(0, self.a_feedback * chart_log_return)
        )
        noise_weight: float = max(
            0, self.w_n - min(0, self.a_noise * chart_log_return)
        )
        weights: list[float] = [self.w_f, chart_weight, noise_weight]
        return weights
