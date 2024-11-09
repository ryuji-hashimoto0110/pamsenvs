import csv
import pathlib
from pathlib import Path
from pams.agents import Agent
from pams.logs.base import SimulationBeginLog, SimulationEndLog
from pams.logs.base import ExecutionLog
from pams.market import Market
from pams.simulator import Simulator
from pams.logs import Logger
from typing import Optional
from typing import TypeVar

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class PortfolioSaver(Logger):
    def __init__(self, dfs_save_path: Path) -> None:
        super().__init__()
        if not dfs_save_path.exists():
            dfs_save_path.mkdir(parents=True)
        self.dfs_save_path: Path = dfs_save_path
        self.market_id2path_dic: dict[MarketID, Path] = {}
        self.agent_id2agent_dic: dict[AgentID, Agent] = {}
        self.market_id2rows_dic: dict[MarketID, list[Optional[str | float | int]]] = {}

    def _create_columns(self) -> list[str]:
         return [
              "time", "agent_id", "execution_price", "execution_volume",
              "holding_cash_amount", "holding_asset_volume"
         ]

    def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
        simulator: Simulator = log.simulator
        for market in simulator.markets:
            market_id: MarketID = market.market_id
            self.market_id2rows_dic[market_id] = [self._create_columns()]
            csv_path: Path = self.dfs_save_path / f"{market_id}.csv"
            self.market_id2path_dic[market_id] = csv_path
            for agent in simulator.agents:
                agent_id: str = agent.agent_id
                self.agent_id2agent_dic[agent_id] = agent
                self.market_id2rows_dic[market_id].append(
                    [
                        -1, agent_id, None, None, agent.cash_amount,
                        agent.asset_volumes[market_id]
                    ]
                )

    def process_execution_log(self, log: ExecutionLog) -> None:
        market_id: MarketID = log.market_id
        t: int = log.time
        buy_agent_id: str = log.buy_agent_id
        buy_agent: Agent = self.agent_id2agent_dic[buy_agent_id]
        sell_agent_id: str = log.sell_agent_id
        sell_agent: Agent = self.agent_id2agent_dic[sell_agent_id]
        execution_price: float = log.price
        execution_volume: int = log.volume
        buy_agent_cash_amount: float = buy_agent.cash_amount - execution_price * execution_volume
        buy_agent_asset_volume: int = buy_agent.asset_volumes[market_id] + execution_volume
        sell_agent_cash_amount: float = sell_agent.cash_amount + execution_price * execution_volume
        sell_agent_asset_volume: int = sell_agent.asset_volumes[market_id] - execution_volume
        self.market_id2rows_dic[market_id].append(
            [
                t, buy_agent_id, execution_price, execution_volume,
                buy_agent_cash_amount, buy_agent_asset_volume
            ]
        )
        self.market_id2rows_dic[market_id].append(
            [
                t, sell_agent_id, execution_price, -execution_volume,
                sell_agent_cash_amount, sell_agent_asset_volume
            ]
        )
            
    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        for market_id, rows in self.market_id2rows_dic.items():
            csv_path: Path = self.market_id2path_dic[market_id]
            with open(csv_path, mode="w") as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
