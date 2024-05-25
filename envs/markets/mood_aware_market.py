from pams.agents import Agent
from pams.logs.base import OrderLog
from pams.order import Order
from .total_time_aware_market import TotalTimeAwareMarket
from pams.logs import Logger
import random
from typing import Any
from typing import Dict
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

    def setup(
        self,
        settings: dict[str, Any],
        *args,
        **kwargs
    ) -> None:
        super().setup(settings, *args, **kwargs)
        if "changeMoodNum" not in settings:
            raise ValueError(
                "MoodAwareMarket requires changeMoodNum in settings."
            )
        self.change_mood_num: int = settings["changeMoodNum"]

    def get_market_mood(self) -> float:
        if len(self.agent_id2mood_dic) == 0:
            return 0.5
        else:
            mood: float = sum(
                list(self.agent_id2mood_dic.values())
            ) / len(self.agent_id2mood_dic)
            return mood
        
    def _add_order(self, order: Order) -> OrderLog:
        self._change_agents_mood()
        agent_id: AgentID = order.agent_id
        agent: Agent = self.simulator.id2agent[agent_id]
        self.agent_id2mood_dic[agent_id] = agent.get_agent_mood()
        return super()._add_order(order)
    
    def _change_agents_mood(self) -> None:
        agent_ids: list[AgentID] = list(self.agent_id2mood_dic.keys())
        agent_ids_to_change: list[AgentID] = self._prng.choices(
            agent_ids, k=min(len(agent_ids), self.change_mood_num)
        )
        for agent_id in agent_ids_to_change:
            agent: Agent = self.simulator.id2agent[agent_id]
            agent.change_mood(self)