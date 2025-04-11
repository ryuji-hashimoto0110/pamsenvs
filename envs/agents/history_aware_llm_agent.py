import json
import math
from pams.logs import ExecutionLog
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from pams.utils import JsonRandom
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
        self.settings: dict[str, Any] = settings
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        if "basePrompt" in settings:
            warnings.warn("basePrompt will be ignored.")
        else:
            self.premise: str = "You are a participant of the simulation of stock markets. " + \
                "Behave as an investor. Answer your order decision after analysing the given information. "
            self.instruction: str = "\\n\\nYour current portfolio is provided as a following format. " + \
                "Cash denotes safe asset. You consume cash to buy stocks, while you gain cash when you sell stocks. " + \
                "Unrealized gain refers to the increase in value of the investment that has not yet been sold. " + \
                "It represents the potential profit on your stock position. Negative unrealized gain means that " + \
                "the investment has decreased in value. " + \
                "\\n[Your portfolio]cash: {}\\n" + \
                "[Your portfolio]market id: {}, volume: {}, unrealized gain: {}, ..." + \
                "\\n\\nEach market condition is provided as a following format." + \
                "\\n[Market condition]market id: {}, current market price: {}, " + \
                "all time high price: {}, all time low price: {}, ..." + \
                "\\n[Market condition]remaining time steps: {} total time steps: {}" + \
                "\\n\\nYour trading history is provided as a following format. " + \
                "Negative volume means that you sold the stock." + \
                "\\n[Your trading history]market id: {}, price: {} volume: {}, ..."
            if not "getOFI" in settings:
                raise ValueError("getOFI must be included in settings.")
            else:
                self.get_ofi: bool = settings["getOFI"]
            if self.get_ofi:
                self.instruction += "\\n\\n Order flow imbalance is provided as a following format. " + \
                    "Order flow imbalance means the difference between the number of buy and sell orders submitted to the stock market. " + \
                    "Order flow imbalance is calculated as the difference between the number of buy and sell orders. " + \
                    "Order flow imbalance can range from -1 to 1. " + \
                    "Negative order flow imbalance indicates that the number of sell orders exceed that of buy orders. " + \
                    "If the order flow is positive (negative), the fundamental value tends to be high (low)." + \
                    "Higher absolute value of order flow imbalance indicates that orders are imbalance to one side, " + \
                    "and suggests stronger evidence about the fundamentals value of the stock." + \
                    "\\n[Order flow imbalance]market id: {}, order flow imbalance: {}, ..."
            self.answer_format: str = "\\n\\nDecide your investment in the following JSON format with its keys are market ids and values are dictionaries " + \
                "Do not deviate from the format, " + \
                "and do not add any additional words to your response outside of the format. " + \
                "Make sure to enclose each property in double quotes. " + \
                "Order volume means the number of units you want to buy or sell the stock. " + \
                "Possible order volume is up to 10. " + \
                "is_buy means whether you want to buy or sell the stock. is_buy must be True or False." + \
                "Short selling is not allowed. Do not sell more stocks than you hold. " + \
                "Try to keep your order volume as non-zero and not-extreme as possible. " + \
                "Order price means the limit price at which you want to buy or sell the stock. By adjusting " + \
                "order price, you can trade at a more favorable price or adjust the time it takes to execute a trade. " + \
                "Here are the answer format." + \
                '\\n{"<market id>": {"order_price": "<order price>", "is_buy": "<True or False>", "order_volume": "<order volume>", "reason": "<reason>"} ...}' + \
                "\\n\\nNow, decide your order. Please explain the reason in as much detail as possible."
            self.base_prompt: str = self.premise + self.instruction
        if not "onlyMarketOrders" in settings:
            raise ValueError("onlyMarketOrders must be included in settings.")
        else:
            self.only_market_orders: bool = settings["onlyMarketOrders"]
        if "orderMargin" in settings:
            self.order_margin: float = json_random.random(
                json_value=settings["orderMargin"]
            )
        else:
            self.order_margin: float = 0.0
        if 0 < len(accessible_markets_ids):
            self.market_id2ofi: dict[MarketID, Optional[float]] = {}
            self.last_reason_dic: dict[MarketID, str] = {}
            self.average_cost_dic: dict[MarketID, float] = {}
        for market_id in accessible_markets_ids:
            self.market_id2ofi[market_id] = None
            self.average_cost_dic[market_id] = 0.0
            self.last_reason_dic[market_id] = ""
    
    def create_ofi_info(self, markets: list[Market]) -> str:
        """create order flow imbalance information."""
        ofi_info: str = ""
        for market in markets:
            market_id: MarketID = market.market_id
            if hasattr(market, "get_ofi"):
                ofi_str, ofi = market.get_ofi()
                self.market_id2ofi[market_id] = ofi
                ofi_info += ofi_str
        return ofi_info
    
    def _create_portfolio_info(self, markets: list[Market]) -> str:
        """create a portfolio information."""
        cash_amount: float = self.get_cash_amount()
        portfolio_info: str = f"\\n[Your portfolio]cash: {cash_amount:.1f}"
        for market in markets:
            market_id: int = market.market_id
            volume: int = self.asset_volumes[market_id]
            unrealized_gain: float = self._get_unrealized_gain(
                market=market, current_volume=volume
            )
            portfolio_info += f"\\n[Your portfolio]market id: {market_id}, " + \
                f"volume: {volume}, unrealized gain: {unrealized_gain:.2f}"
        return portfolio_info
    
    def _get_unrealized_gain(
        self,
        market: Market,
        current_volume: int
    ) -> float:
        market_id: int = market.market_id
        execution_logs: list[ExecutionLog] = self.executed_orders_dic[market_id]
        total_shares: int = 0
        total_cost: float = 0.0
        for log in execution_logs:
            if log.buy_agent_id == self.agent_id:
                total_shares += log.volume
                total_cost += log.price * log.volume
            elif log.sell_agent_id == self.agent_id:
                total_shares -= log.volume
                total_cost -= log.price * log.volume
            else:
                raise ValueError(
                    "Unrelated execution log found in executed_orders_dic."
                )
        if total_shares == 0:
            self.average_cost_dic[market_id] = 0.0
            return 0.0
        current_price: float = market.get_market_price()
        average_cost: float = total_cost / total_shares
        self.average_cost_dic[market_id] = average_cost
        unrealized_gain: float = (current_price - average_cost) * total_shares
        return unrealized_gain
    
    def _create_market_condition_info(self, markets: list[Market]) -> str:
        """create a market condition information."""
        market_condition_info: str = ""
        for market in markets:
            market_id: int = market.market_id
            current_market_price: float = market.get_market_price()
            market_prices: list[float] = market.get_market_prices()
            if hasattr(market, "all_time_high"):
                all_time_high_price: float = market.all_time_high
                all_time_low_price: float = market.all_time_low
            else:
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
                f"\\n[Market condition]remaining time step: {remaining_time} total time steps: {total_time}"
        return market_condition_info
    
    def _create_trading_history_info(self) -> str:
        """create a trading history information."""
        trading_history_info: str = ""
        for market_id in self.executed_orders_dic.keys():
            execution_logs: list[ExecutionLog] = self.executed_orders_dic[market_id]
            count = 0
            for execution_log in reversed(execution_logs):
                if count >= 5:
                    break
                price: float = execution_log.price
                volume: float = execution_log.volume
                if execution_log.buy_agent_id == self.agent_id:
                    volume = volume
                elif execution_log.sell_agent_id == self.agent_id:
                    volume = -volume
                else:
                    raise ValueError("The agent id does not match the buy agent id or the sell agent id.")
                trading_history_info += f"\n[Your trading history]market id: {market_id}, " + \
                    f"price: {price:.1f}, volume: {volume:.1f}"
                count += 1
        return trading_history_info
    
    def create_prompt(self, markets: list[Market]) -> str:
        """create a prompt for the agent."""
        portfolio_info: str = self._create_portfolio_info(markets=markets)
        market_condition_info: str = self._create_market_condition_info(markets=markets)
        trading_history_info: str = self._create_trading_history_info()
        prompt: str = self.base_prompt + "\\n Here are the information." + portfolio_info + \
            market_condition_info + trading_history_info
        ofi_info: str = self.create_ofi_info(markets=markets)
        if self.get_ofi:
            prompt += ofi_info
        prompt += self.answer_format
        prompt = json.dumps({"text": prompt}, ensure_ascii=False)
        return prompt
        
    def convert_llm_output2orders(
        self,
        llm_output: str,
        markets: list[Market],
        exo_order_price_dic: Optional[dict[MarketID, float]] = None,
        exo_order_volume_dic: Optional[dict[MarketID, int]] = None
    ) -> list[Order | Cancel]:
        """convert the LLM output to orders."""
        orders_dic: dict[MarketID, dict] = json.loads(llm_output)
        orders: list[Order | Cancel] = []
        for market_id, order_dic in orders_dic.items():
            if not "order_volume" in order_dic:
                raise ValueError("order_volume must be included in order_dic.")
            else:
                order_volume: str = order_dic["order_volume"]
            try:
                market_id = int(market_id)
            except ValueError:
                raise ValueError(f"Failed to convert market_id to an integer {market_id}.")
            try:
                order_volume = abs(int(float(order_volume)))
            except ValueError:
                raise ValueError(f"Failed to convert order_volume to an integer: {order_volume}.")
            if not "reason" in order_dic:
                reason: str = ""
            else:
                reason: str = order_dic["reason"]
            self.last_reason_dic[market_id] = reason
            if order_volume == 0:
                order_volume += 1
            if self.only_market_orders:
                order_kind = MARKET_ORDER
                order_price = None
            else:
                if not "order_price" in order_dic:
                    raise ValueError("order_price must be included in order_dic.")
                else:
                    order_kind = LIMIT_ORDER
                    order_price: Any = order_dic["order_price"]
                    try:
                        order_price = float(order_price)
                    except ValueError:
                        raise ValueError(f"Failed to convert order_price to a float: {order_price}.")
            is_buy: bool = True if order_dic["is_buy"].lower() == "true" else False
            if exo_order_price_dic is not None:
                order_kind = LIMIT_ORDER
                order_price: float = exo_order_price_dic[market_id]
                prder_price = order_price * (1.0 - self.order_margin) if is_buy else \
                    order_price * (1.0 + self.order_margin)
                market: Market = markets[market_id]
                if is_buy:
                    best_ask: float = market.get_best_sell_price()
                    if best_ask is not None:
                        order_price = min(order_price, best_ask)
                else:
                    best_bid: float = market.get_best_buy_price()
                    if best_bid is not None:
                        order_price = max(order_price, best_bid)
            if exo_order_volume_dic is not None:
                order_volume: int = exo_order_volume_dic[market_id]
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


class HistoryAwareFCLAgent(HistoryAwareLLMAgent):
    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.w_f: float = json_random.random(json_value=settings["fundamentalWeight"])
        self.w_c: float = json_random.random(json_value=settings["chartWeight"])
        self.w_n: float = json_random.random(json_value=settings["noiseWeight"])
        self.noise_scale: float = json_random.random(json_value=settings["noiseScale"])
        self.time_window_size = int(
            json_random.random(json_value=settings["timeWindowSize"])
        )
        self.mean_reversion_time: int = int(
            json_random.random(json_value=settings["meanReversionTime"])
        )

    def get_exo_order_prices_volumes(
        self,
        markets: list[Market]
    ) -> tuple[dict[MarketID, float], dict[MarketID, int]]:
        exo_order_price_dic: dict[MarketID, float] = {}
        exo_order_volume_dic: dict[MarketID, int] = {}
        for market in markets:
            market_id: MarketID = market.market_id
            expected_future_price: float = self._get_expected_future_price(
                market=market
            )
            exo_order_price_dic[market_id] = expected_future_price
            exo_order_volume_dic[market_id] = 1
        return exo_order_price_dic, exo_order_volume_dic
    
    def _get_expected_future_price(self, market: Market) -> float:
        time: int = market.get_time()
        market_price: float = market.get_market_price()
        fundamental_price: float = market.get_fundamental_price()
        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return: float = fundamental_scale * math.log(
            fundamental_price / market_price
        )
        chart_scale: float = 1.0 / max(self.time_window_size, 1)
        time_window_size: int = min(time, self.time_window_size)
        chart_log_return: float = chart_scale * math.log(
            market_price / market.get_market_price(time - time_window_size)
        )
        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        expected_log_return: float = (
            1.0 / (self.w_f + self.w_c + self.w_n)
        ) * (
            self.w_f * fundamental_log_return
            + self.w_c * chart_log_return + self.w_n * noise_log_return
        )
        expected_future_price: float = market_price * math.exp(
            expected_log_return * time_window_size
        )
        return expected_future_price
