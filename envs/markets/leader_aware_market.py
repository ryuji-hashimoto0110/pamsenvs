from pams.agents import Agent
from pams.logs import Logger
from pams.logs import OrderLog
from pams.market import Market
from pams.order import Order
from pams.order_book import OrderBook
import pathlib
from pathlib import Path
import random
from .total_time_aware_market import TotalTimeAwareMarket
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeVar

AgentID = TypeVar("AgentID")

class LeaderAwareMarket(TotalTimeAwareMarket): 
    """Leader-aware Market class."""
    def setup(
        self,
        settings: dict[str, Any],
        *args,
        **kwargs
    ) -> None:
        """market setup.
        
        LeaderAwareMarket provides text-format private signal and leaderboard to each agent.
        
        Args:
            settings (dict[str, Any]): market configuration. This must include the parameters in addition to tickSize, marketPrice / fundamentalPrice:
                    consistentSignalRate (float): The probability that the market privides private
                        signal that is consistent with the fundamental value.
                    signalsPath (str): The path to the signals folder.
                    devidendPrice (float): The price of the devidend per unit.
                    averageStockValue (float): The value of one unit of the stock. This is used to calculate wealth.
        """
        super().setup(settings, *args, **kwargs)
        if not "consistentSignalRate" in settings:
            raise ValueError("consistentSignalRate is required for LeaderAwareMarket setting.")
        else:
            self.consistent_signal_rate: float = settings["consistentSignalRate"]
        if not "devidendPrice" in settings:
            raise ValueError("devidendPrice is required for LeaderAwareMarket setting.")
        else:
            self.devidend_price: float = settings["devidendPrice"]
        if not "signalsPath" in settings:
            raise ValueError("signalsPath is required for LeaderAwareMarket setting.")
        else:
            self.signals_path: Path = pathlib.Path(settings["signalsPath"]).resolve()
            if not self.signals_path.exists():
                raise FileNotFoundError(f"Folder not found: {self.signals_path}")
        self.overweight_txt_paths: list[Path] = self._collect_signal_paths(overweight=True)
        self.underweight_txt_paths: list[Path] = self._collect_signal_paths(overweight=False)
        if not "averageStockValue" in settings:
            raise ValueError("averageStockValue is required for LeaderAwareMarket setting.")
        else:
            self.average_stock_value: float = settings["averageStockValue"]
        self.leader2wealth_dic: dict[AgentID, float] = {}
        self.leader2action_dic: dict[AgentID, Literal["buy", "sell"]] = {}
        self.agent2wealth_dic: dict[AgentID, float] = {}
        self.overweight_rate: float = 0.0
        self.num_buy_orders: int = 0
        self.num_sell_orders: int = 0
    
    def _collect_signal_paths(self, overweight: bool = True) -> list[Path]:
        if overweight:
            return sorted(
                list(self.signals_path.rglob("*promotion_overweight.txt"))
            )
        else:
            return sorted(
                list(self.signals_path.rglob("*promotion_underweight.txt"))
            )
        
    def _calc_wealth(self, agent: Agent) -> float:
        cash_amount: float = agent.cash_amount
        asset_volume: int = agent.asset_volumes[self.market_id]
        remaining_time: int = self.get_remaining_time()
        wealth: float = cash_amount + asset_volume * self.average_stock_value * remaining_time
        return wealth

    def _update_time(self, next_fundamental_price: float) -> None:
        super()._update_time(next_fundamental_price)
        for agent in self.simulator.agents:
            self.agent2wealth_dic[agent.agent_id] = self._calc_wealth(agent)

    def init_session(self) -> None:
        """initialize session. This is called by LeaderAwareMarketInitializer.

        This method initializes the session by the following procedure.
            1. Clear buy/sell order book.
            2. initialize OFI.
            3. Uniformly sample the devidend from {0, self.devidend_price}.
            4. set overweight and underweight signals.
            5. Get top-3 agents and register them as leaders.
        """
        self._clear_order_book(self.buy_order_book)
        self._clear_order_book(self.sell_order_book)
        self.num_buy_orders = 0
        self.num_sell_orders = 0
        self.devidend = self._prng.choice([0, self.devidend_price])
        if self.devidend == self.devidend_price:
            self.overweight_rate: float = self.consistent_signal_rate
        elif self.devidend == 0:
            self.overweight_rate: float = 1 - self.consistent_signal_rate
        else:
            raise ValueError(f"Invalid devidend: {self.devidend}")
        self.overweight_signal: str = self._get_signal_fromt_txt(self.overweight_txt_paths)
        self.underweight_signal: str = self._get_signal_fromt_txt(self.underweight_txt_paths)
        self._update_leaderboard()

    def _get_signal_fromt_txt(self, txt_paths: list[Path]) -> str:
        if len(txt_paths) == 0:
            raise ValueError("signals exhausted.")
        txt_path: Path = txt_paths[0]
        del txt_paths[0]
        with txt_path.open("r") as f:
            private_signal: str = f.read()
        return private_signal

    def _clear_order_book(self, order_book: OrderBook) -> None:
        for order in order_book.priority_queue:
            order_book._remove(order)

    def _update_leaderboard(self) -> None:
        self.leader2wealth_dic = {}
        sorted_wealths: list[float] = sorted(
            list(self.agent2wealth_dic.values()), reverse=True
        )
        while len(self.leader2wealth_dic) < 3:
            for wealth in sorted_wealths:
                agent_ids: list[AgentID] = [
                    agent_id for agent_id, w in self.agent2wealth_dic.items() if w == wealth
                ]
                for agent_id in agent_ids:
                    if not agent_id in self.leader2wealth_dic:
                        self.leader2wealth_dic[agent_id] = wealth
                    if 3 <= len(self.leader2wealth_dic):
                        break

    def get_leaderboard(self) -> str:
        """get leaderboard."""
        leaderboard: str = ""
        wealthes: list[float] = sorted(
            list(self.leader2wealth_dic.values()), reverse=True
        )
        rank: int = 0
        for wealth in wealthes:
            agent_ids: list[AgentID] = [
                agent_id for agent_id, w in self.leader2wealth_dic.items() if w == wealth
            ]
            for agent_id in agent_ids:
                rank += 1
                if agent_id in self.leader2action_dic:
                    action: str = self.leader2action_dic[agent_id]
                else:
                    action: str = "None"
                leaderboard += f"[Leaderboard]rank: {rank}th, wealth: {wealth}, action: {action}\\n"
        return leaderboard

    def get_ofi(self) -> str:
        if self.num_buy_orders + self.num_sell_orders == 0:
            ofi: float = 0.0
        else:
            ofi: float = (
                self.num_buy_orders - self.num_sell_orders
            ) / (
                self.num_buy_orders + self.num_sell_orders
            )
        return f"[Order Flow Imbalance]{ofi}\\n"

    def _add_order(self, order: Order) -> OrderLog:
        log: OrderLog = super()._add_order(order)
        is_buy: bool = order.is_buy
        if is_buy:
            self.num_buy_orders += 1
        else:
            self.num_sell_orders += 1
        agent_id: AgentID = order.agent_id
        if agent_id in self.leader2wealth_dic:
            if is_buy:
                self.leader2action_dic[agent_id] = "buy"
            else:
                self.leader2action_dic[agent_id] = "sell"

