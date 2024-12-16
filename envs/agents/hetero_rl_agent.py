from pams.agents import Agent
from pams.logs import OrderLog
from pams.market import Market
from pams.order import Order
from pams.order import Cancel
from pams.utils import JsonRandom
from typing import Any

class HeteroRLAgent(Agent):
    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids,
        *args,
        **kwargs
    ):
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.last_order_time: int = -1
        if "skillBoundedness" not in settings:
            raise ValueError("skillBoundedness is required for HeteroRLAgent.")
        else:
            self.skill_boundedness: float = json_random.random(
                json_value=settings["skillBoundedness"]
            )
        if "riskAversionTerm" not in settings:
            raise ValueError("riskAversionTerm is required for HeteroRLAgent.")
        else:
            self.risk_aversion_term: float = json_random.random(
                json_value=settings["riskAversionTerm"]
            )
        if "discountFactor" not in settings:
            raise ValueError("discountFactor is required for HeteroRLAgent.")
        else:
            self.discount_factor: float = json_random.random(
                json_value=settings["discountFactor"]
            )

    def submit_orders(self, markets: list[Market]) -> list[Order | Cancel]:
        return []
    
    def submitted_order(self, log: OrderLog) -> None:
        self.last_order_time = log.time