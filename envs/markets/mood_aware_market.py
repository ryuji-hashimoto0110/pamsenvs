from pams.agents import Agent
from pams.logs.base import OrderLog
from pams.order import Order
from .total_time_aware_market import TotalTimeAwareMarket
from pams.logs import Logger
import random
from typing import Literal
from typing import Optional
from typing import TypeVar

AgentID = TypeVar("AgentID")

class MoodAwareMarket(TotalTimeAwareMarket):
    """Mood-Aware Market class.

    MoodAwareMarket requires all agents to be MoodAwareCARAFCNAgent.
    In other words, agent called in this market must have method: ".get_agent_mood()"
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
        self.agent_id2mood_dic: dict[AgentID, Literal[0, 1]] = {}

    def get_market_mood(self) -> float:
        if len(self.agent_id2mood_dic) == 0:
            return 0.5
        else:
            mood: float = sum(
                list(self.agent_id2mood_dic.values())
            ) / len(self.agent_id2mood_dic)
            return mood
        
    def _add_order(self, order: Order) -> OrderLog:
        agent_id: AgentID = order.agent_id
        agent: Agent = self.simulator.id2agent[agent_id]
        self.agent_id2mood_dic[agent_id] = agent.get_agent_mood()
        return super()._add_order(order)