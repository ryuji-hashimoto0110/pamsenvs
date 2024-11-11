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
    def __init__(
        self,
        dfs_save_path: Path,
        record_ofi: bool = False,
        record_leader_board: bool = False,
        record_signal_description: bool = False
    ) -> None:
        super().__init__()
        if not dfs_save_path.exists():
            dfs_save_path.mkdir(parents=True)
        self.dfs_save_path: Path = dfs_save_path
        self.market_id2path_dic: dict[MarketID, Path] = {}
        self.agent_id2agent_dic: dict[AgentID, Agent] = {}
        self.market_id2rows_dic: dict[MarketID, list[Optional[str | float | int]]] = {}
        self.record_ofi: bool = record_ofi
        self.record_leader_board: bool = record_leader_board
        self.record_signal_description: bool = record_signal_description

    def _create_columns(self) -> list[str]:
        column_names: list[str] = [
            "time", "agent_id", "execution_price", "execution_volume",
            "holding_cash_amount", "holding_asset_volume", "reason"
        ]
        if self.record_signal_description:
            column_names.extend(["devidend_price", "private_signal"])
        if self.record_ofi:
            column_names.append("ofi")
        if self.record_leader_board:
            column_names.extend(
                [
                    "leader1_id", "leader1_wealth", "leader1_action",
                    "leader2_id", "leader2_wealth", "leader2_action",
                    "leader3_id", "leader3_wealth", "leader3_action"
                ]
            )

    def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
        simulator: Simulator = log.simulator
        for market in simulator.markets:
            market_id: MarketID = market.market_id
            csv_path: Path = self.dfs_save_path / f"market{market_id}.csv"
            self.market_id2rows_dic[market_id] = [self._create_columns()]
            self.market_id2path_dic[market_id] = csv_path
        for agent in simulator.agents:
            agent_id: AgentID = agent.agent_id
            self.agent_id2agent_dic[agent_id] = agent

    def process_execution_log(self, log: ExecutionLog) -> None:
        market_id: MarketID = log.market_id
        buy_agent_id: str = log.buy_agent_id
        buy_agent: Agent = self.agent_id2agent_dic[buy_agent_id]
        sell_agent_id: str = log.sell_agent_id
        sell_agent: Agent = self.agent_id2agent_dic[sell_agent_id]
        buy_agent_infos: list[Optional[str | float | int]] = self._record_agent_infos(
            log, buy_agent
        )
        sell_agent_infos: list[Optional[str | float | int]] = self._record_agent_infos(
            log, sell_agent
        )
        self.market_id2rows_dic[market_id].append(buy_agent_infos)
        self.market_id2rows_dic[market_id].append(sell_agent_infos)

    def _record_agent_infos(
        self,
        log: ExecutionLog,
        agent: Agent
    ) -> list[Optional[str | float | int]]:
        market_id: MarketID = log.market_id
        t: int = log.time
        execution_price: float = log.price
        execution_volume: int = log.volume
        agent_id: AgentID = agent.agent_id
        agent_cash_amount: float = agent.cash_amount
        agent_asset_volume: int = agent.asset_volumes[market_id]
        if hasattr(agent, "last_reason_dic"):
            reason: str = agent.last_reason_dic[market_id]
        else:
            reason: str = None
        agent_infos: list[Optional[str | float | int]] = [
            t, agent_id, execution_price, execution_volume,
            agent_cash_amount, agent_asset_volume, reason
        ]
        if self.record_signal_description:
            if hasattr(agent, "market_id2signal_descriptions"):
                signal_descriptions: list[Optional[str]] = agent.market_id2signal_descriptions[market_id]
                agent_infos.extend(signal_descriptions)
            else:
                agent_infos.extend([None, None])
        if self.record_ofi:
            if hasattr(agent, "market_id2ofi"):
                ofi: float = agent.market_id2ofi[market_id]
                agent_infos.append(ofi)
            else:
                agent_infos.append(None)
        if self.record_leader_board:
            if hasattr(agent, "market_id2lb"):
                lb: list[Optional[int | str | float]] = agent.market_id2lb[market_id]
                agent_infos.extend(lb)
            else:
                agent_infos.extend([None for _ in range(9)])
        return agent_infos
            
    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        for market_id, rows in self.market_id2rows_dic.items():
            csv_path: Path = self.market_id2path_dic[market_id]
            with open(csv_path, mode="w") as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
