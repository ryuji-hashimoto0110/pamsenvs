from abc import ABC, abstractmethod
from gymnasium import Space
import numpy as np
from pams.agents import Agent
from pams.agents import HighFrequencyAgent
from pams.logs import Log
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Log
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.logs import SessionBeginLog
from pams.logs import SessionEndLog
from pams.logs import SimulationBeginLog
from pams.logs import SimulationEndLog
from pams.market import Market
from pams.order import Cancel
from pams.order import Order
from pams.runners import Runner
from pams.session import Session
from pams.simulator import Simulator
from pettingzoo import AECEnv
import random
from random import Random
import torch
from typing import Optional, TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
InfoType = TypeVar("InfoType")
MarketID = TypeVar("MarketID")
ObsType = TypeVar("ObsType")

class PamsAECEnv(AECEnv, ABC):
    """PamsAECEnv class.

    Multi-agent environment class for pams.
    PamsAECEnv treat pams based artificial market as Agent Environment Cycle (AEC) environment.
    """
    def __init__(
        self,
        config_dic: dict,
        variable_ranges_dic: Optional[dict],
        simulator_class: type[Simulator],
        target_agent_names: list[str],
        action_dim: int,
        obs_dim: int,
        logger: Optional[Logger] = None,
        seed: Optional[int] = None
    ) -> None:
        """initialization.

        Args:
            config_dic (dict): runner configuration. (=settings)
            variable_ranges_dic (Optional[dict]): dict to specify the ranges of values for variables in config.
                Ex: {"Market1": {"fundamentalDrift": [-0.001,0.001], "fundamentalVolatility": [0,0.0001]},
                    ...}
                The values of variables are sampled by .modify_config method as each episode.
            simulator_class (type[Simulator]): type of simulator.
            target_agent_names (list[str]): target Agent names.
            action_dim (int): dimension of action spaces.
            obs_dim (int): dimension of observation spaces.
            logger (Optional[Logger]): logger instance. Defaults to None.
        """
        self.config_dic: dict = config_dic
        self.variable_ranges_dic: Optional[dict] = variable_ranges_dic
        self.simulator_class: type[Simulator] = simulator_class
        self.target_agent_names: list[str] = target_agent_names
        self.action_dim: list[int] = action_dim
        self.obs_dim: list[int] = obs_dim
        self.logger: Optional[Logger] = logger
        self._prng: Random = random.Random()
        self.action_space: Space = self.set_action_space()
        self.obs_space: Space = self.set_obs_space()
        if seed is not None:
            self.seed(seed)
        self.agent_selection: AgentID

    def last(self) -> ObsType:
        """_summary_

        Returns:
            ObsType: 
        """
        agent_id: AgentID = self.agent_selection
        obs: ObsType = self.generate_obs(agent_id)
        return obs

    @abstractmethod
    def set_action_space(self) -> Space:
        pass

    @abstractmethod
    def set_obs_space(self) -> Space:
        pass

    def seed(self, seed: int) -> None:
        """set seed.

        Args:
            seed (int): seed value.
        """
        self._prng.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms = True

    def get_time(self) -> int:
        if hasattr(self, "markets"):
            return self.markets[0].get_time()
        else:
            return -1

    def reset(self) -> None:
        """reset environment.

        initialize and set up runner and send initial observation to target agent.

        This method
            1. modify config file according to randomlly generate variables by refering to variable_ranges_dic.
            2. initialize and setup runner.
            3. add optional attributes to the environment.
            4. iterate markets untill target agent is called for submitting orders.

        - self.current_session_time is current number of time steps within the session.

        Returns:
            obs (ObsType): initial observation.
        """
        initial_config_dic: dict = self.config_dic.copy()
        episode_config_dic: dict = self.modify_config(
            initial_config_dic, self.variable_ranges_dic
        )
        self.runner: Runner = self.setup_runner(episode_config_dic, self.logger)
        self.simulator: Simulator = self.runner.simulator
        self.agents: list[AgentID] = [
            self.simulator.name2agent[name].agent_id for name in self.target_agent_names
        ]
        self.sessions: list[Session] = self.simulator.sessions
        self.markets: list[Market] = self.simulator.markets
        self._start_simulation()
        target_agent_id, unplaced_local_orders, done = self.iterate_markets_til_target_agent_is_called()
        if done:
            self.reset()
        assert target_agent_id is not None and unplaced_local_orders is not None
        self.agent_selection: AgentID = target_agent_id
        self.unplaced_local_orders: list[list[Order | Cancel]] = unplaced_local_orders

    def step(
        self,
        action: ActionType
    ) -> tuple[float, bool, InfoType]:
        """step environment.

        receive action by the agent and step the environment.
        This method
            1. convert action (probably np.ndarray) to list of orders.
            2. handle orders if the target agent is hft agent, add orders to the local orders otherwise.
            3. handle remaining local orders. If hft agent is called to take actions,
               return reward, done, and info and wait for the action of the called hft agent.
            4. collect local orders until one of the target agents is called for submitting orders.

        Args:
            action (ActionType): action by target agent.

        Returns:
            reward (float):
            done (bool):
            info (InfoType):
        """
        target_orders: list[Order | Cancel] = self.convert_action2orders(action)
        target_agent_id: AgentID = self.agent_selection
        if isinstance(
            self.simulator.id2agent[target_agent_id], HighFrequencyAgent
        ):
            self.handle_orders_by_single_agent(target_orders)
        else:
            self.unplaced_local_orders.append(target_orders)
        self.unplaced_local_orders, target_agent_id_ = \
            self.handle_orders_wo_target_agents(
                session=self.current_session,
                unplaced_local_orders=self.unplaced_local_orders
            )
        if target_agent_id_ is not None:
            self.agent_selection = target_agent_id_
            reward: float = self.generate_reward(target_agent_id)
            done: bool = False
            info: InfoType = self.generate_info(target_agent_id)
            return reward, done, info
        if len(self.unplaced_local_orders) != 0:
            raise ValueError(
                "unplaced orders remain, even though order handling completed."
            )
        target_agent_id_, unplaced_local_orders, done = \
            self.iterate_markets_til_target_agent_is_called()
        if target_agent_id_ is None or unplaced_local_orders is None:
            if not done:
                raise ValueError(
                    "target_agent_id_ or unplaced_local_orders is None, even though episode did not end."
                )
        self.agent_selection = target_agent_id_
        self.unplaced_local_orders: list[list[Order | Cancel]] = unplaced_local_orders
        reward: float = self.generate_reward(target_agent_id)
        info: InfoType = self.generate_info(target_agent_id)
        return reward, done, info
    
    def _start_simulation(self) -> None:
        self.current_session_idx: int = 0
        self.current_session_time: int = 0
        if self.logger is not None:
            log: Log = SimulationBeginLog(simulator=self.simulator)
            log.read_and_write_with_direct_process(logger=self.logger)
        self.simulator._update_times_on_markets(self.markets)
        self.add_attributes()
        self.n_orders: int = 0
        self.n_hft_orders: int = 0
    
    def _step_simulation(self) -> None:
        for market in self.markets:
            if self.logger is not None:
                log = MarketStepEndLog(
                    session=self.current_session, market=market, simulator=self.simulator
                )
                log.read_and_write_with_direct_process(logger=self.logger)
            self.simulator._trigger_event_after_step_for_market(market=market)
        self.simulator._update_times_on_markets(self.markets)
        self.current_session_time += 1
        self.n_orders = 0
        self.n_hft_orders = 0

    def _start_session(self) -> None:
        self.simulator._trigger_event_before_session(session=self.current_session)
        if self.logger is not None:
            log: Log = SessionBeginLog(
                session=self.current_session, simulator=self.simulator
            )
            log.read_and_write_with_direct_process(logger=self.logger)

    def _step_session(self) -> bool:
        self.simulator._trigger_event_after_session(session=self.current_session)
        done: bool = False
        if self.logger is not None:
            log = SessionEndLog(
                session=self.current_session, simulator=self.simulator
            )
            log.read_and_write_with_direct_process(logger=self.logger)
        if self.current_session_idx + 1 == len(self.sessions):
            done: bool = True
            if self.logger is not None:
                log = SimulationEndLog(simulator=self.simulator)
                log.read_and_write_with_direct_process(logger=self.logger)
        else:
            self.current_session_idx += 1
            self.current_session_time = 0
        return done

    def iterate_markets_til_target_agent_is_called(
        self
    ) -> tuple[Optional[AgentID], Optional[list[list[Order | Cancel]]], bool]:
        """iterate markets until one of the target agents is called for submitting orders.

        Returns:
            agent_id (AgentID, optional): id of called target agent.
            unplaced_local_orders (list[list[Order | Cancel]], optinal):
                local orders that have not yet placed at markets.
            done (bool): whether the simulation ended or not.
        """
        done: bool = False
        while True:
            self.simulator.current_session = self.sessions[self.current_session_idx]
            self.current_session: Session = self.simulator.current_session
            is_ending_session: bool = False
            if self.current_session_time == 0:
                self._start_session()
            elif self.current_session_time == self.current_session.iteration_steps-1:
                is_ending_session = True
            if self.n_orders == 0:
                for market in self.markets:
                    market._is_running = self.current_session.with_order_execution
                    self.simulator._trigger_event_before_step_for_market(market=market)
                    if self.logger is not None:
                        log = MarketStepBeginLog(
                            session=self.current_session, market=market, simulator=self.simulator
                        )
                        log.read_and_write_with_direct_process(logger=self.logger)
            if self.current_session.with_order_placement:
                unplaced_local_orders, target_agent_id = \
                    self.collect_orders_from_normal_agents_wo_target_agent(
                        session=self.current_session
                    )
                if target_agent_id is not None:
                    return target_agent_id, unplaced_local_orders, done
                unplaced_local_orders, target_agent_id = \
                    self.handle_orders_wo_target_agents(
                        session=self.current_session, unplaced_local_orders=unplaced_local_orders
                    )
                if target_agent_id is not None:
                    return target_agent_id, unplaced_local_orders, done
            if self.n_orders <= self.current_session.max_normal_orders:
                self._step_simulation()
                if is_ending_session:
                    done = self._step_session()
                    if done:
                        return None, None, done

    def collect_orders_from_normal_agents_wo_target_agent(
        self,
        session: Session
    ) -> tuple[list[list[Order | Cancel]], Optional[AgentID]]:
        """_summary_

        Args:
            session (Session): _description_

        Returns:
            unplaced_local_orders (list[list[Order | Cancel]]): _description_
            is_target_agent_called (bool): _description_
        """
        agents: list[Agent] = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        unplaced_local_orders: list[list[Order | Cancel]] = []
        target_agent_id: Optional[AgentID] = None
        for agent in agents:
            if session.max_normal_orders <= self.n_orders:
                break
            if agent.name in self.target_agent_names:
                self.n_orders += 1
                target_agent_id: AgentID = agent.agent_id
                break
            else:
                orders: list[Order | Cancel] = agent.submit_orders(
                    markets=self.simulator.markets
                )
                if len(orders) > 0:
                    unplaced_local_orders.append(orders)
                    self.n_orders += 1
        return unplaced_local_orders, target_agent_id

    def handle_orders_wo_target_agents(
        self,
        session: Session,
        unplaced_local_orders: list[list[Order | Cancel]]
    ) -> tuple[list[list[Order | Cancel]], Optional[AgentID]]:
        """_summary_

        Args:
            session (Session): _description_
            unplaced_local_orders (list[list[Order | Cancel]]): _description_

        Returns:
            unplaced_local_orders (list[list[Order | Cancel]]): _description_
            is_target_agent_called (bool): _description_
        """
        removing_orders: list[list[Order | Cancel]] = []
        target_agent_id: Optional[AgentID] = None
        for orders in unplaced_local_orders:
            self.handle_orders_by_single_agent(session, orders)
            removing_orders.append(orders)
            if session.high_frequency_submission_rate < self._prng.random():
                continue
            agents = self.simulator.high_frequency_agents
            agents = self._prng.sample(agents, len(agents))
            for agent in agents:
                if session.max_high_frequency_orders <= self.n_hft_orders:
                    break
                if agent.name in self.target_agent_names:
                    self.n_hft_orders += 1
                    target_agent_id: AgentID = agent.agent_id
                    break
                high_freq_orders: list[Order | Cancel] = agent.submit_orders(
                    markets=self.simulator.markets
                )
                self.n_hft_orders += 1
                self.handle_orders_by_single_agent(
                    session, high_freq_orders
                )
            if target_agent_id is not None:
                break
        for orders in removing_orders:
            unplaced_local_orders.remove(orders)
        return unplaced_local_orders, target_agent_id

    def handle_orders_by_single_agent(
        self,
        session: Session,
        orders: list[Order | Cancel]
    ) -> None:
        for order in orders:
            market: Market = self.simulator.id2market[order.market_id]
            if isinstance(order, Order):
                self.simulator._trigger_event_before_order(order=order)
                log: OrderLog = market._add_order(order=order)
                agent: Agent = self.simulator.id2agent[order.agent_id]
                agent.submitted_order(log=log)
                self.simulator._trigger_event_after_order(order_log=log)
            elif isinstance(order, Cancel):
                self.simulator._trigger_event_before_cancel(cancel=order)
                log_: CancelLog = market._cancel_order(cancel=order)
                agent = self.simulator.id2agent[order.order.agent_id]
                agent.canceled_order(log=log_)
                self.simulator._trigger_event_after_cancel(cancel_log=log_)
            if session.with_order_execution:
                logs: list[ExecutionLog] = market._execution()
                self.simulator._update_agents_for_execution(execution_logs=logs)
                for execution_log in logs:
                    agent = self.simulator.id2agent[execution_log.buy_agent_id]
                    agent.executed_order(log=execution_log)
                    agent = self.simulator.id2agent[execution_log.sell_agent_id]
                    agent.executed_order(log=execution_log)
                    self.simulator._trigger_event_after_execution(
                        execution_log=execution_log
                    )

    @abstractmethod
    def modify_config(
        self,
        initial_config_dic: dict,
        variable_ranges_dic: dict
    ) -> dict:
        pass

    @abstractmethod
    def setup_runner(
        self,
        episode_config_dic: dict,
        logger: Optional[Logger] = None
    ) -> Runner:
        pass

    def add_attributes(self) -> None:
        pass

    @abstractmethod
    def generate_obs(self, agent_id: AgentID) -> ObsType:
        pass

    @abstractmethod
    def generate_reward(self, agent_id: AgentID) -> float:
        pass

    @abstractmethod
    def generate_info(self, agent_id: AgentID) -> InfoType:
        pass

    @abstractmethod
    def convert_action2orders(self, action: ActionType) -> list[Order | Cancel]:
        pass

    @abstractmethod
    def is_ready_to_store_experience(self) -> bool:
        pass
