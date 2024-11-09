import json
from pams.logs.base import ExecutionLog
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from .history_aware_llm_agent import HistoryAwareLLMAgent
from typing import Any
from typing import Optional
from typing import TypeVar
import warnings
from rich import print

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class LeaderAwareLLMAgent(HistoryAwareLLMAgent):
    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """agent setup.
        
        Args:
            settings (dict[str, Any]): agent configuration. This must include the parameters:
                cashAmount (float): the initial cash amount.
                assetVolume (int): the initial asset volume.
                llmName (str): the name of the language model.
                devidendPrice (float): The price of the devidend per unit.
                getOFI (bool): whether to get order flow imbalance.
                getLeaderBoard (bool): whether to get leader board.
            and can include the parameter:
                base_prompt (str): the base prompt for the agent.
            accessible_markets_ids (list[MarketID]): list of accessible market ids.
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        if not "devidendPrice" in settings:
            raise ValueError("devidendPrice must be included in settings.")
        else:
            devidend_price: float = settings["devidendPrice"]
        if not "getOFI" in settings:
            raise ValueError("getOFI must be included in settings.")
        else:
            self.get_ofi: bool = settings["getOFI"]
        if not "getLeaderBoard" in settings:
            raise ValueError("getLeaderBoard must be included in settings.")
        else:
            self.get_lb: bool = settings["getLeaderBoard"]
        self.premise += "At each time steps, you can receive devidend as cash according to your holding stock volume: " + \
            "your cash amount <- your cash amount +  asset volume * devidend price. " + \
            f"The devidend price is decided to be 0 or {devidend_price} according to the fundamental value of the stock. " + \
            "You should buy the stock if you believe that the fundamental value is high to gain devidend. " + \
            "Your goal is to achieve high wealth. Your wealth is calculated as: " + \
            "your cash amount + sum(asset volume * (average stock value * remaining time steps)). " + \
            "Note that stocks will become less valuable as the time goes by." 
        if self.get_ofi:
            self.instruction += "\\n\\n Order flow imbalance is provided as a following format. " + \
                "Order flow imbalance means the difference between the number of buy and sell orders submitted to the stock market. " + \
                "Negative order flow imbalance indicates that the number of sell orders exceed that of buy orders. " + \
                "Higher absolute value of order flow imbalance indicates that orders are imbalance to one side." + \
                "\\n[Order flow imbalance]market id: {}, order flow imbalance: {}, ..."
        if self.get_lb:
            self.instruction += "\\n\\n Leader board is provided as a following format." + \
                "\\n[Leaderboard]market id: {}, rank: {}, wealth: {}, order direction: {}, ..."
        self.instruction += "\\n\\nIn addition to the above information, " + \
            "private signal is provided. Private signal is a description about the fundamental value of the stock. " + \
            "If the private signal seems to be overweighted, the fundamental value tends to be high. " + \
            "Leaderboard is provided as a following format.\\n [Private signal]market id: {}, private signal: {}, ..."
        self.base_prompt: str = self.premise + self.instruction

    def create_ofi_info(self, markets: list[Market]) -> str:
        """create order flow imbalance information."""
        ofi_info: str = ""
        for market in markets:
            if hasattr(market, "get_ofi"):
                ofi_info += market.get_ofi()
        return ofi_info
    
    def create_lb_info(self, markets: list[Market]) -> str:
        """create leader board information."""
        lb_info: str = ""
        for market in markets:
            if hasattr(market, "get_leaderboard"):
                lb_info += market.get_leaderboard()
        return lb_info
    
    def create_private_signal_info(self, markets: list[Market]) -> str:
        """create private signal information."""
        private_signal_info: str = ""
        for market in markets:
            if hasattr(market, "get_private_signal"):
                private_signal_info += market.get_private_signal()
        return private_signal_info
        
    def create_prompt(self, markets: list[Market]) -> str:
        """create a prompt for the agent."""
        cash_amount: float = self.get_cash_amount()
        portfolio_info: str = self._create_portfolio_info()
        market_condition_info: str = self._create_market_condition_info(markets=markets)
        trading_history_info: str = self._create_trading_history_info()
        prompt: str = self.base_prompt + "\\n Here are the information." + portfolio_info + \
            market_condition_info + trading_history_info 
        if self.get_ofi:
            prompt += self.create_ofi_info(markets=markets)
        if self.get_lb:
            prompt += self.create_lb_info(markets=markets)
        prompt += self.create_private_signal_info(markets=markets)
        prompt += self.answer_format
        prompt = json.dumps({"text": prompt}, ensure_ascii=False)
        print("[green]==prompt==[green]")
        print(prompt)
        print()
        return prompt