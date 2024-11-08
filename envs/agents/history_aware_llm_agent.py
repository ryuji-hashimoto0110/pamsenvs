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
            premise: str = "You are an individual investor in stock markets. " + \
                "Decide whether to buy or sell stocks to each market or do nothing.\n\n"
            instruction: str = "Your current portfolio is provided as a following format.\n" + \
                "[Your portfolio]cash: {}\n" + \
                "[Your portfolio]market id: {}, volume: {}\n\n" +\
                "Each market condition is provided as a following format.\n" + \
                "[Market condition]market id: {}, current market price: {}, " + \
                "all time high price: {}, all time low price: {}\n\n" + \
                "Your trading history is also provided as a following format. " + \
                "Negative volume means that you sold the stock.\n" + \
                "[Your trading history]market id: {}, price: {} volume: {}\n\n"
            answer_format: str = "Please answer in the following dictionary format. " + \
                "Do not deviate from the format, " + \
                "and do not add any additional words to your response outside of the format. " + \
                "order volume means the number of units you want to buy or sell the stock. " + \
                "Negative order volume means that you want to sell the stock. " + \
                "If you want to do nothing, please do not include the market id in your response. " + \
                "Short selling is not allowed.\n" + \
                "{{market id}: {order volume}, {market id}: {order volume}, ...}\n\n"
            self.base_prompt: str = premise + instruction + answer_format
    
    def _create_portfolio_info(self) -> str:
        """create a portfolio information."""
        cash_amount: float = self.get_cash_amount()
        portfolio_info: str = f"[Your portfolio]cash: {cash_amount:.1f}\n"
        for market_id, volume in self.asset_volumes.items():
            portfolio_info += f"[Your portfolio]market id: {market_id}, volume: {volume}\n"
        return portfolio_info
    
    def _create_market_condition_info(self, markets: list[Market]) -> str:
        """create a market condition information."""
        market_condition_info: str = ""
        for market in markets:
            market_id: int = market.market_id
            current_market_price: float = market.get_market_price()
            market_prices: list[float] = market.get_market_prices()
            all_time_high_price: float = max(market_prices)
            all_time_low_price: float = min(market_prices)
            market_condition_info += f"[Market condition]market id: {market_id}, " + \
                f"current market price: {current_market_price:.1f}, " + \
                f"all time high price: {all_time_high_price:.1f}, " + \
                f"all time low price: {all_time_low_price:.1f}\n"
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
                    f"price: {price:.1f}, volume: {volume:.1f}\n"
        return trading_history_info
    
    def create_prompt(self, markets: list[Market]) -> str:
        """create a prompt for the agent."""
        cash_amount: float = self.get_cash_amount()
        portfolio_info: str = self._create_portfolio_info()
        market_condition_info: str = self._create_market_condition_info(markets=markets)
        trading_history_info: str = self._create_trading_history_info()
        prompt: str = self.base_prompt + portfolio_info + market_condition_info + trading_history_info
        print(prompt)
        
    def convert_llm_output2orders(
        self,
        llm_output: str,
        market: Market
    ) -> list[Order | Cancel]:
        """convert the LLM output to orders."""
        print(llm_output)
        success: bool = False
        for _ in range(10):
            try:
                order_dic: dict[MarketID, int] = json.loads(llm_output)
                success = True
            except json.JSONDecodeError:
                continue
            if success:
                break
        if not success:
            raise ValueError("Failed to convert the LLM output to a dictionary.")
        orders: list[Order | Cancel] = []
        for market_id, order_volume in order_dic.items():
            try:
                market_id = int(market_id)
            except ValueError:
                raise ValueError(f"The market id must be an integer but found {market_id}.")
            try:
                order_volume = int(order_volume)
            except ValueError:
                raise ValueError(f"Failed to convert order_volume to an integer: {order_volume}.")
            if order_volume == 0:
                continue
            if order_volume < 0:
                is_buy: bool = False
                order_volume = - order_volume
                order = Order(
                    agent_id=self.agent_id,
                    market_id=market_id,
                    is_buy=is_buy,
                    price=None,
                    volume=order_volume,
                    kind=MARKET_ORDER,
                    ttl=1
                )
            else:
                is_buy: bool = True
                order = Order(
                    agent_id=self.agent_id,
                    market_id=market_id,
                    is_buy=is_buy,
                    price=None,
                    volume=order_volume,
                    kind=MARKET_ORDER,
                    ttl=1
                )
            orders.append(order)
        return orders      
            