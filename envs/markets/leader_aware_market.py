from pams.agents import Agent
from pams.logs import Logger
from pams.logs.base import OrderLog
from pams.market import Market
from pams.order import Order
from pams.order_book import OrderBook
import pathlib
from pathlib import Path
import random
from .total_time_aware_market import TotalTimeAwareMarket
from typing import Any
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
                    isOrderFlowImbalanceAvailable (bool): Whether the market provides order flow imbalance or not.
                    isLeaderBoardAvailable (bool): Whether the market provides leaderboard or not.
                    devidendPrice (float): The price of the devidend per unit.
                    averageStockValue (float): The value of one unit of the stock. This is used to calculate wealth.
        """
        super().setup(settings, *args, **kwargs)
        if not "consistentSignalRate" in settings:
            raise ValueError("consistentSignalRate is required for LeaderAwareMarket setting.")
        else:
            self.consistent_signal_rate: float = settings["consistentSignalRate"]
        if not "isOrderFlowImbalanceAvailable" in settings:
            raise ValueError("isOrderFlowImbalanceAvailable is required for LeaderAwareMarket setting.")
        else:
            self.is_ofi_available: bool = settings["isOrderFlowImbalanceAvailable"]
        if not "isLeaderBoardAvailable" in settings:
            raise ValueError("isLeaderBoardAvailable is required for LeaderAwareMarket setting.")
        else:
            self.is_lb_available: bool = settings["isLeaderBoardAvailable"]
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
        self.agent2wealth_dic: dict[AgentID, float] = {}
        self.overweight_rate: float = 0.0
    
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
            2. Uniformly sample the devidend from {0, self.devidend_price}.
            3. set overweight and underweight signals.
            4. Get top-3 agents and register them as leaders.
        """
        self._clear_order_book(self.buy_order_book)
        self._clear_order_book(self.sell_order_book)
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

