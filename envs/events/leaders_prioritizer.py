import random
from ..markets import LeaderAwareMarket
from pams.agents import Agent
from pams.events import EventABC
from pams.events import EventHook
from pams.session import Session
from pams.simulator import Simulator
from typing import Literal
from typing import TypeVar

AgentID = TypeVar("AgentID")

class LeadersPrioritizer(EventABC):
    def __init__(
        self,
        event_id: int,
        prng: random.Random,
        session: Session,
        simulator: Simulator,
        name: str
    ) -> None:
        super().__init__(event_id, prng, session, simulator, name)
        self.normal_frequency_agents: list[Agent] = simulator.normal_frequency_agents
        self.called_agent_ids: list[AgentID] = []

    def hook_registration(self) -> list[EventHook]:
        self.start_time: int = self.session.session_start_time
        self.end_time: int = self.start_time + 3
        event_hook = EventHook(
            event=self,
            hook_type="session",
            time=[self.start_time + t for t in range(4)],
            is_before=True
        )
        return [event_hook]
    
    def hooked_before_step_for_market(
        self,
        simulator: Simulator,
        market: LeaderAwareMarket,
    ) -> None:
        """initialize LeaderAwareMarket.
        """
        current_time: int = market.get_time()
        if current_time == self.start_time-1:
            print(f"{current_time} LeadersPrioritizer: pick top-1.")
            leaders: list[Agent] = self.pick_leader(market, 1)
        elif current_time == self.start_time:
            print(f"{current_time} LeadersPrioritizer: pick top-2.")
            leaders: list[Agent] = self.pick_leader(market, 2)
        elif current_time == self.start_time+1:
            print(f"{current_time} LeadersPrioritizer: pick top-3.")
            leaders: list[Agent] = self.pick_leader(market, 3)
        elif current_time == self.end_time-1:
            print(f"{current_time} LeadersPrioritizer: pick others.")
            leaders: list[Agent] = simulator.normal_frequency_agents
        else:
            raise ValueError(f"Invalid time: {current_time=}, [{self.start_time}, {self.end_time}].")
        simulator.normal_frequency_agents = leaders

    def pick_leader(
        self,
        market: LeaderAwareMarket,
        rank: Literal[1, 2, 3]
    ) -> list[Agent]:
        """pick leader agent.
        """
        leader2wealth_dic: dict[AgentID, float] = market.leader2wealth_dic
        if len(leader2wealth_dic) == 0:
            raise ValueError(
                "leader2wealth_dic is empty. Run LeaderAwareMarket.init_session() first."
            )
        sorted_wealths: list[float] = sorted(list(leader2wealth_dic.values()), reverse=True)
        target_wealth: float = sorted_wealths[rank - 1]
        for agent_id, wealth in leader2wealth_dic.items():
            if wealth == target_wealth:
                if not agent_id in self.called_agent_ids:
                    self.called_agent_ids.append(agent_id)
                    return [self.simulator.id2agent[agent_id]]
                else:
                    continue