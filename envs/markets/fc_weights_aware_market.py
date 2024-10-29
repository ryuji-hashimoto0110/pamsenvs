from pams.agents import Agent
from pams.logs import Logger
from pams.logs.base import OrderLog
from pams.market import Market
from pams.order import Order
import random
from .range_regulated_market import RangeRegulatedMarket
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import TypeVar

AgentID = TypeVar("AgentID")

class FCWeightsAwareMarket(RangeRegulatedMarket):
    """F, C and time_window_size-aware Market class.
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
        self.agent_id2wf_dic: dict[AgentID, float] = {}
        self.agent_id2wc_dic: dict[AgentID, float] = {}
        self.wc_rate: float = 0.0
        self.previous_time_window_size: int = 0

    def _update_time(self, next_fundamental_price: float) -> None:
        super()._update_time(next_fundamental_price)
        if self.time == 1:
            for agent in self.simulator.agents:
                if hasattr(agent, "w_f") and hasattr(agent, "w_c"):
                    self.agent_id2wf_dic[agent.agent_id] = agent.w_f
                    self.agent_id2wc_dic[agent.agent_id] = agent.w_c
        
    def _add_order(self, order: Order) -> OrderLog:
        agent_id: AgentID = order.agent_id
        agent: Agent = self.simulator.id2agent[agent_id]
        if hasattr(agent, "w_f") and hasattr(agent, "w_c"):
            self.agent_id2wf_dic[agent_id] = agent.w_f
            self.agent_id2wc_dic[agent_id] = agent.w_c
            self.previous_time_window_size = agent._calc_temporal_time_window_size(
                time=self.time,
                fundamental_weight=agent.w_f, chart_weight=agent.w_c,
                market=self
            )
        total_wc: float = sum(list(self.agent_id2wc_dic.values()))
        total_wf: float = sum(list(self.agent_id2wf_dic.values()))
        if not total_wc + total_wf == 0:
            self.wc_rate = total_wc / (total_wc + total_wf)
        return super()._add_order(order)