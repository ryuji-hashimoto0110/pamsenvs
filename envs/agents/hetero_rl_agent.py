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
        self.last_order_time: int = 0
        self.num_executed_orders: int = 0
        if "skillBoundedness" not in settings:
            raise ValueError("skillBoundedness is required for HeteroRLAgent.")
        else:
            self.skill_boundedness: float = json_random.random(
                json_value=settings["skillBoundedness"]
            )
            if "averageCashAmount" in settings:
                average_cash_amount: float = settings["averageCashAmount"]
                cash_amount: float = self.cash_amount
                #self.skill_boundedness *= average_cash_amount / cash_amount
        if "riskAversionTerm" not in settings:
            raise ValueError("riskAversionTerm is required for HeteroRLAgent.")
        else:
            self.risk_aversion_term: float = json_random.random(
                json_value=settings["riskAversionTerm"]
            )
            if "averageCashAmount" in settings:
                average_cash_amount: float = settings["averageCashAmount"]
                cash_amount: float = self.cash_amount
                #self.risk_aversion_term *= average_cash_amount / cash_amount
        if "discountFactor" not in settings:
            raise ValueError("discountFactor is required for HeteroRLAgent.")
        else:
            self.discount_factor: float = json_random.random(
                json_value=settings["discountFactor"]
            )
        self.previous_utility: float = self.cash_amount
        for market_id in accessible_markets_ids:
            market: Market = self.simulator.id2market[market_id]
            asset_volume: int = self.asset_volumes[market_id]
            market_price: float = market.get_market_price()
            self.previous_utility += asset_volume * market_price

    def submit_orders(self, markets: list[Market]) -> list[Order | Cancel]:
        return []
    
    def submitted_order(self, log: OrderLog) -> None:
        self.last_order_time = log.time
        self.num_executed_orders = 0

    def executed_order(self, log):
        self.num_executed_orders += log.volume