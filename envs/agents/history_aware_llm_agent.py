import json
from pams.logs.base import ExecutionLog
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from .prompt_aware_agent import PromptAwareAgent
from typing import Any
from typing import Optional
from typing import TypeVar
import warnings
from rich import print

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class HistoryAwareLLMAgent(PromptAwareAgent):
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
                onlyMarketOrders (bool): whether to submit market orders only.
            and can include the parameter:
                base_prompt (str): the base prompt for the agent.
            accessible_markets_ids (list[MarketID]): list of accessible market ids.
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        if "basePrompt" in settings:
            warnings.warn("basePrompt will be ignored.")
        else:
            self.premise: str = "This is a social experiment in a laboratory setting. " + \
                "Behave as an investor in stock markets. Answer your order decision by analysing the given information."
            self.instruction: str = "\\n\\nYour current portfolio is provided as a following format." + \
                "\\n[Your portfolio]cash: {}\\n" + \
                "[Your portfolio]market id: {}, volume: {}" + \
                "\\n\\nEach market condition is provided as a following format." + \
                "\\n[Market condition]market id: {}, current market price: {}, " + \
                "all time high price: {}, all time low price: {}" + \
                "\\n[Market condition]remaining time step: {}/{}" + \
                "\\n\\nYour trading history is also provided as a following format. " + \
                "Negative volume means that you sold the stock." + \
                "\\n[Your trading history]market id: {}, price: {} volume: {}" + \
                "\\n[Your trading history]market id: {}, price: {} volume: {} ..."
            self.answer_format: str = "\\n\\nDecide your investment in the following JSON format. " + \
                "Do not deviate from the format, " + \
                "and do not add any additional words to your response outside of the format. " + \
                "Make sure to enclose each property in double quotes." + \
                '\\n{<market id>: {"order_price": <order price>, "order_volume": <order volume>, "reason": <reason>} ...}' + \
                "\\n\\norder volume means the number of units you want to buy or sell the stock. " + \
                "Negative order volume means that you want to sell the stock. " + \
                "Short selling is not allowed."
            self.base_prompt: str = self.premise + self.instruction
        if not "onlyMarketOrders" in settings:
            raise ValueError("onlyMarketOrders must be included in settings.")
        else:
            self.only_market_orders: bool = settings["onlyMarketOrders"]
    
    def _create_portfolio_info(self) -> str:
        """create a portfolio information."""
        cash_amount: float = self.get_cash_amount()
        portfolio_info: str = f"\\n[Your portfolio]cash: {cash_amount:.1f}"
        for market_id, volume in self.asset_volumes.items():
            portfolio_info += f"\\n[Your portfolio]market id: {market_id}, volume: {volume}\\n"
        return portfolio_info
    
    def _create_market_condition_info(self, markets: list[Market]) -> str:
        """create a market condition information."""
        market_condition_info: str = ""
        for market in markets:
            market_id: int = market.market_id
            current_market_price: float = market.get_fundamental_price()
            market_prices: list[float] = market.get_fundamental_prices()
            all_time_high_price: float = max(market_prices)
            all_time_low_price: float = min(market_prices)
            if hasattr(market, "get_remaining_time"):
                remaining_time: int = market.get_remaining_time()
                current_time: int = market.get_time()
                total_time: int = current_time + remaining_time
            else:
                raise ValueError("The market does not have the method get_remaining_time.")
            market_condition_info += f"\\n[Market condition]market id: {market_id}, " + \
                f"current market price: {current_market_price:.1f}, " + \
                f"all time high price: {all_time_high_price:.1f}, " + \
                f"all time low price: {all_time_low_price:.1f}" + \
                f"\\n[Market condition]remaining time step: {remaining_time}/{total_time}"
        return market_condition_info
    
    def _create_trading_history_info(self) -> str:
        """create a trading history information."""
        trading_history_info: str = ""
        for market_id in self.executed_orders_dic.keys():
            execution_logs: list[ExecutionLog] = self.executed_orders_dic[market_id]
            for execution_log in execution_logs:
                price: float = execution_log.price
                volume: float = execution_log.volume
                if execution_log.buy_agent_id == self.agent_id:
                    volume = volume
                elif execution_log.sell_agent_id == self.agent_id:
                    volume = -volume
                else:
                    raise ValueError("The agent id does not match the buy agent id or the sell agent id.")
                trading_history_info += f"[Your trading history]market id: {market_id}, " + \
                    f"price: {price:.1f}, volume: {volume:.1f}\\n"
        trading_history_info += "\\n"
        return trading_history_info
    
    def create_prompt(self, markets: list[Market]) -> str:
        """create a prompt for the agent."""
        portfolio_info: str = self._create_portfolio_info()
        market_condition_info: str = self._create_market_condition_info(markets=markets)
        trading_history_info: str = self._create_trading_history_info()
        prompt: str = self.base_prompt + portfolio_info + \
            market_condition_info + trading_history_info + self.answer_format
        prompt = json.dumps({"text": prompt}, ensure_ascii=False)
        return prompt
        
    def convert_llm_output2orders(
        self,
        llm_output: str,
        markets: list[Market]
    ) -> list[Order | Cancel]:
        """convert the LLM output to orders."""
        success: bool = False
        order_dic: dict[MarketID, int] = json.loads(llm_output)
        print(order_dic)
        print()
        orders: list[Order | Cancel] = []
        for market_id, order_dic in order_dic.items():
            if not "order_volume" in order_dic:
                raise ValueError("order_volume must be included in order_dic.")
            else:
                order_volume: Any = order_dic["order_volume"]
            try:
                market_id = int(market_id)
            except ValueError:
                raise ValueError(f"Failed to convert market_id to an integer {market_id}.")
            try:
                order_volume = int(order_volume)
            except ValueError:
                raise ValueError(f"Failed to convert order_volume to an integer: {order_volume}.")
            if order_volume == 0:
                continue
            if self.only_market_orders:
                order_kind = MARKET_ORDER
                order_price = None
            else:
                if not "order_price" in order_dic:
                    raise ValueError("order_price must be included in order_dic.")
                else:
                    order_kind = MARKET_ORDER
                    order_price: Any = order_dic["order_price"]
                    try:
                        order_price = float(order_price)
                    except ValueError:
                        raise ValueError(f"Failed to convert order_price to a float: {order_price}.")
            if order_volume < 0:
                is_buy: bool = False
                order_volume = - order_volume
            else:
                is_buy: bool = True
            order = Order(
                agent_id=self.agent_id,
                market_id=market_id,
                is_buy=is_buy,
                price=order_price,
                volume=order_volume,
                kind=order_kind
            )
            orders.append(order)
        return orders      
            