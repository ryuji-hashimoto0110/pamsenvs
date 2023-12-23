from abc import ABC, abstractmethod
from gym import Env
from gym import Space
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
import random
from random import Random
import torch
from typing import Optional, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
InfoType = TypeVar("InfoType")

class PamsEnv(Env, ABC):
    """PamsEnv class.

    Single agent RL environment for an agent in PAMS.
    This class inherits from the gym.Env class.
    """
    def __init__(
        self,
        config_dic: dict,
        variable_ranges_dic: Optional[dict],
        simulator_class: type[Simulator],
        target_agent_name: str,
        action_dim: int,
        obs_dim: int,
        logger: Optional[Logger] = None,
    ) -> None:
        """initialization.

        Args:
            config_dic (dict): runner configuration. (=settings)
            variable_ranges_dic (Optional[dict]): dict to specify the ranges of values for variables in config.
                Ex: {"Market1": {"fundamentalDrift": [-0.001,0.001], "fundamentalVolatility": [0,0.0001]},
                    ...}
                The values of variables are sampled by .modify_config method as each episode.
            simulator_class (type[Simulator]): type of simulator.
            target_agent_name (str): target Agent name.
            action_dim (int): dimension of action space.
            obs_dim (int): dimension of observation space.
            logger (Optional[Logger]): logger instance. Defaults to None.
        """
        self.config_dic: dict = config_dic
        self.variable_ranges_dic: Optional[dict] = variable_ranges_dic
        self.simulator_class: type[Simulator] = simulator_class
        self.target_agent_name: str = target_agent_name
        self.action_dim: int = action_dim
        self.obs_dim: int = obs_dim
        self.logger: Optional[Logger] = logger
        self._prng: Random = random.Random()
        self.action_space: Space = self.set_action_space()
        self.obs_space: Space = self.set_obs_space()

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
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    def reset(self) -> ObsType:
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
        self.target_agent: Agent = self.simulator.name2agent[self.target_agent_name]
        self.is_hft: bool = isinstance(self.target_agent, HighFrequencyAgent)
        self.sessions: list[Session] = self.simulator.sessions
        self.markets: list[Market] = self.simulator.markets
        self.current_session_idx: int = 0
        self.current_session_time: int = 0
        if self.logger is not None:
            log: Log = SimulationBeginLog(simulator=self.simulator)
            log.read_and_write_with_direct_process(logger=self.logger)
        self.add_attributes()
        self.unplaced_local_orders, _ = self.iterate_markets_til_target_agent_is_called()
        obs: ObsType = self.generate_obs()
        return obs

    def step(
        self,
        action: ActionType
    ) -> tuple[ObsType, float, bool, InfoType]:
        """step environment.

        receive action by the agent and step the environment.

        Args:
            action (ActionType): action by target agent.

        Returns:
            next_obs (ObsType):
            reward (float):
            done (bool):
            info (InfoType):
        """
        target_orders: list[Order | Cancel] = self.convert_action2orders(action)
        if self.is_hft:
            self.handle_orders_by_single_agent(target_orders)
            self.unplaced_local_orders, is_target_agent_called = \
                self.handle_orders_wo_target_agents(
                    session=self.current_session,
                    unplaced_local_orders=self.unplaced_local_orders
                )
            if is_target_agent_called:
                next_obs: ObsType = self.generate_obs()
                reward: float = self.generate_reward()
                done: bool = False
                info: InfoType = self.generate_info()
                return next_obs, reward, done, info
        else:
            self.unplaced_local_orders.append(target_orders)
            self.runner._handle_orders(
                session=self.current_session, local_orders=self.unplaced_local_orders
            )
        for market in self.markets:
            if self.logger is not None:
                log = MarketStepEndLog(
                    session=self.current_session, market=market, simulator=self.simulator
                )
                log.read_and_write_with_direct_process(logger=self.logger)
            self.simulator._trigger_event_after_step_for_market(market=market)
        self.unplaced_local_orders, done = self.iterate_markets_til_target_agent_is_called()
        next_obs: ObsType = self.generate_obs()
        reward: float = self.generate_reward()
        info: InfoType = self.generate_info()
        return next_obs, reward, done, info

    def iterate_markets_til_target_agent_is_called(
        self
    ) -> tuple[Optional[list[list[Order | Cancel]]], bool]:
        """iterate markets until target agent is called for submitting orders.

        Returns:
            unplaced_local_orders (Optional[list[list[Order | Cancel]]]):
                local orders that have not yet placed at markets.
            done (bool): whether the simulation ended or not.
        """
        done: bool = False
        while True:
            self.simulator._update_times_on_markets(self.markets)
            self.market_time: int = self.markets[0].get_time()
            self.simulator.current_session = self.sessions[self.current_session_idx]
            self.current_session: Session = self.simulator.current_session
            if self.current_session_time == 0:
                self.simulator._trigger_event_before_session(session=self.current_session)
                if self.logger is not None:
                    log: Log = SessionBeginLog(
                        session=self.current_session, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
                self.current_session_time += 1
            elif self.current_session_time == self.current_session.iteration_steps:
                self.simulator._trigger_event_after_session(session=self.current_session)
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
                    return None, done
                self.current_session_idx += 1
                self.current_session_time = 0
            else:
                self.current_session_time += 1
            for market in self.markets:
                market._is_running = self.current_session.with_order_execution
                self.simulator._trigger_event_before_step_for_market(market=market)
                if self.logger is not None:
                    log = MarketStepBeginLog(
                        session=self.current_session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
                if self.current_session.with_order_placement:
                    if self.is_hft:
                        unplaced_local_orders: list[list[Order | Cancel]] = \
                            self.runner._collect_orders_from_normal_agents(
                                session=self.current_session
                            )
                        unplaced_local_orders, is_target_agent_called = \
                            self.handle_orders_wo_target_agents(
                                session=self.current_session,
                                unplaced_local_orders=unplaced_local_orders
                            )
                        if is_target_agent_called:
                            return unplaced_local_orders, done
                    else:
                        unplaced_local_orders, is_target_agent_called = \
                            self.collect_orders_from_normal_agents_wo_target_agent(
                                session=self.current_session
                            )
                        if is_target_agent_called:
                            return unplaced_local_orders, done
                        self.runner._handle_orders(
                            session=self.current_session, local_orders=unplaced_local_orders
                        )
            for market in self.markets:
                if self.logger is not None:
                    log = MarketStepEndLog(
                        session=self.current_session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
                self.simulator._trigger_event_after_step_for_market(market=market)

    def collect_orders_from_normal_agents_wo_target_agent(
        self,
        session: Session
    ) -> tuple[list[list[Order | Cancel]], bool]:
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
        n_orders: int = 0
        is_target_agent_called: bool = False
        for agent in agents:
            if session.max_normal_orders <= n_orders:
                break
            if agent.name == self.target_agent_name:
                n_orders += 1
                is_target_agent_called = True
                continue
            orders: list[Order | Cancel] = agent.submit_orders(markets=self.simulator.markets)
            if len(orders) > 0:
                unplaced_local_orders.append(orders)
                n_orders += 1
        return unplaced_local_orders, is_target_agent_called

    def handle_orders_wo_target_agents(
        self,
        session: Session,
        unplaced_local_orders: list[list[Order | Cancel]]
    ) -> tuple[list[list[Order | Cancel]], bool]:
        """_summary_

        Args:
            session (Session): _description_
            unplaced_local_orders (list[list[Order | Cancel]]): _description_

        Returns:
            unplaced_local_orders (list[list[Order | Cancel]]): _description_
            is_target_agent_called (bool): _description_
        """
        removing_orders: list[list[Order | Cancel]] = []
        is_target_agent_called: bool = False
        for orders in enumerate(unplaced_local_orders):
            self.handle_orders_by_single_agent(
                session, orders
            )
            removing_orders.append(orders)
            if session.high_frequency_submission_rate < self._prng.random():
                continue
            n_high_freq_orders: int = 0
            agents = self.simulator.high_frequency_agents
            agents = self._prng.sample(agents, len(agents))
            for agent in agents:
                if n_high_freq_orders >= session.max_high_frequency_orders:
                    break
                if agent.name == self.target_agent_name:
                    is_target_agent_called = True
                    break
                high_freq_orders: list[Order | Cancel] = agent.submit_orders(markets=self.simulator.markets)
                n_high_freq_orders += 1
                self.handle_orders_by_single_agent(
                    session, high_freq_orders
                )
            if is_target_agent_called:
                break
        for orders in removing_orders:
            unplaced_local_orders.remove(orders)
        return unplaced_local_orders, is_target_agent_called

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
    def generate_obs(self) -> ObsType:
        pass

    @abstractmethod
    def generate_reward(self) -> float:
        pass

    @abstractmethod
    def generate_info(self) -> float:
        pass

    @abstractmethod
    def convert_action2orders(self, action: ActionType) -> list[Order | Cancel]:
        pass
