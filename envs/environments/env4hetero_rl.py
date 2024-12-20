from ..agents import HeteroRLAgent
from .environment import PamsAECEnv
from ..events import DividendProviderwEverySteps
from gymnasium import spaces
from gymnasium import Space
from ..markets import TotalTimeAwareMarket
import numpy as np
from numpy import ndarray
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import Order
from pams.runners import Runner
from pams.runners import SequentialRunner
from pams.simulator import Simulator
from typing import Optional
from typing import TypeVar

ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
InfoType = TypeVar("InfoType")
MarketID = TypeVar("MarketID")
ObsType = TypeVar("ObsType")

class AECEnv4HeteroRL(PamsAECEnv):
    def __init__(
        self,
        config_dic: dict,
        variable_ranges_dic: Optional[dict],
        simulator_class: type[Simulator],
        target_agent_names: list[str],
        action_dim: int,
        obs_dim: int,
        logger: Optional[Logger] = None,
        seed: Optional[int] = None,
        depth_range: float = 0.01,
        limit_order_range: float = 0.1,
        max_order_volume: int = 10,
        short_selling_penalty: float = 1e+03,
        negative_utility_penality: float = 1e+03
    ) -> None:
        """initialization.

        Args:
            depth_range (float): Depth range. The number of limit orders with prices
                between the current price p_t plus and minus depth_range * p_t is used to calculate
                depth features.
            limit_order_range (float): Limit order range. Each agent can place limit orders within the
                range of [p_t - limit_order_range * p_t, p_t + limit_order_range * p_t].
            max_order_volume (int): Maximum order volume.
        """
        super().__init__(
            config_dic=config_dic,
            variable_ranges_dic=variable_ranges_dic,
            simulator_class=simulator_class,
            target_agent_names=target_agent_names,
            action_dim=action_dim,
            obs_dim=obs_dim,
            logger=logger,
            seed=seed
        )
        self.variables_dic: dict[str, dict[str, float]] = {}
        self.depth_range: float = depth_range
        self.limit_order_range: float = limit_order_range
        self.max_order_volume: int = max_order_volume
        self.short_selling_penalty: float = short_selling_penalty
        self.negative_utility_penality: float = negative_utility_penality

    def set_action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    def set_obs_space(self) -> Space:
        return spaces.Box(low=-1e+06, high=1e+06, shape=(self.obs_dim,))

    def modify_config(
        self,
        initial_config_dic: dict,
        variable_ranges_dic: dict
    ) -> dict:
        """modify config_dic.

        uniformelly sample the value of variables at each episode during training.

        Args:
            initial_config_dic (dict): initial config dic.
            variable_ranges_dic (dict): dic to specify the ranges of values for variables in config.

        Returns:
            episode_config_dic (dict): episode config dic. each variables are uniformlly sampled.

        Ex)
        initial_config_dic: dict =
                            {"simulation": {...},
                            "Market1": {..., "fundamentalPrice": 300, "fundamentalDrift": 0.0, "fundamentalVolatility": 0.001, ...},
                            ...}
        variable_ranges_dic: dict =
                            {"Market1": {"fundamentalPrice": [100,300], "fundamentalDrift": [-0.2, 0.2], "fundamentalVolatility": [0., 0.1]},
                            ...}
        """
        episode_config_dic: dict = initial_config_dic.copy()
        key1: str
        key2: str
        for key1 in variable_ranges_dic.keys():
            for key2 in variable_ranges_dic[key1].keys():
                variable_range_list: list = variable_ranges_dic[key1][key2]
                if not isinstance(variable_range_list, list):
                    raise ValueError(
                        f"variable_ranges_dic[{key1}][{key2}] is not variable range list, but {variable_range_list}."
                    )
                if len(variable_range_list) != 2:
                    raise ValueError(
                        f"len(variable_ranges_dic[{key1}][{key2}]) must be 2 but is {len(variable_range_list)}."
                    )
                episode_config_dic[key1][key2] = \
                    self._prng.uniform(variable_range_list[0], variable_range_list[1])
                if key1 not in self.variables_dic.keys():
                    self.variables_dic[key1] = {key2: episode_config_dic[key1][key2]}
                else:
                    self.variables_dic[key1][key2] = episode_config_dic[key1][key2]
        return episode_config_dic

    def setup_runner(
        self,
        episode_config_dic: dict,
        logger: Optional[Logger] = None
    ) -> Runner:
        runner: Runner = SequentialRunner(
            settings=episode_config_dic,
            logger=logger,
            prng=self._prng
        )
        self._register_classes(runner)
        runner._setup()
        simulator: Simulator = runner.simulator
        if len(simulator.markets) != 1:
            raise NotImplementedError("Only one market is supported.")
        return runner

    def _register_classes(self, runner: Runner) -> None:
        runner.class_register(TotalTimeAwareMarket)
        runner.class_register(HeteroRLAgent)
        runner.class_register(DividendProviderwEverySteps)

    """
    def _get_agent_traits_dic(
        self,
        runner: Runner
    ) -> dict[AgentID, dict[str, float | dict[MarketID, float]]]:
        get agent traits.

        'agent traits' are the attributes of agents that are used to generate 
        observations and rewards, containing:
            - last order time
            - holding asset volume
            - holding cash amount
            - skill boundedness
            - risk aversion term
            - discount factor
        simulator: Simulator = runner.simulator
        agent_traits_dic: dict[AgentID, dict[str, float]] = {}
        for agent in simulator.agents:
            agent_id: AgentID = agent.agent_id
            agent_traits_dic[agent_id] = {
                "holding_cash_amount": agent.cash_amount,
                "holding_asset_volume": agent.asset_volumes.copy()
            }
            if hasattr(agent, "last_order_time"):
                agent_traits_dic[agent_id]["last_order_time"] = agent.last_order_time
            else:
                raise ValueError(f"agent {agent_id} does not have last_order_time.")
            if hasattr(agent, "skill_boundedness"):
                agent_traits_dic[agent_id]["skill_boundedness"] = agent.skill_boundedness
            else:
                raise ValueError(f"agent {agent_id} does not have skill_boundedness.")
            if hasattr(agent, "risk_aversion_term"):
                agent_traits_dic[agent_id]["risk_aversion_term"] = agent.risk_aversion_term
            else:
                raise ValueError(f"agent {agent_id} does not have risk_aversion_term.")
            if hasattr(agent, "discount_factor"):
                agent_traits_dic[agent_id]["discount_factor"] = agent.discount_factor
            else:
                raise ValueError(f"agent {agent_id} does not have discount_factor.")
        return agent_traits_dic
    """

    def add_attributes(self) -> None:
        self.return_dic: dict[AgentID, float] = {}
        self.volatility_dic: dict[AgentID, float] = {}
        for agent_id in self.agents:
            self.return_dic[agent_id] = 0.0
            self.volatility_dic[agent_id] = 0.0

    def generate_obs(self, agent_id: AgentID) -> ObsType:
        """generate observation at time t.
        
        Observation of each agent is consists of:
            - Percentage of the agent's total wealth accounted for by the asset.
            - holding asset volume / max order volume.
            - market price / holding cash amount.
            - (T-t) / T where T is the total number of time steps in the episode.
            - return log p_t - log p_{t-tau} where tau is the time interval between the current and previous observations.
            - volatility V_[t-tau, t] where tau is the time interval between the current and previous observations.
            - Percentage of the number of buy and sell limit orders with prices between [p_t-0.01p_t, p_t+0.01p_t]
                accounted for by the agent's holding asset volume.
                This amount can serve as the indicator of market liquidity and order flow imbalance.
            - log difference log p_t^f - log p_t blurred according to the agent's skill boundedness.
            - the agent's skill boundedness.
            - the agent's risk aversion term.
            - the agent's discount factor.
        
        Args:
            agent_id (AgentID): agent id.
        
        Returns:
            obs (ObsType): observation.
        """
        agent: HeteroRLAgent = self.simulator.agents[agent_id]
        market: TotalTimeAwareMarket = self.simulator.markets[0]
        asset_ratio: float = self._calc_asset_ratio(agent, market)
        liquidable_asset_ratio: float = self._liquidable_asset_ratio(agent, market)
        inverted_buying_power: float = self._calc_inverted_buying_power(agent, market)
        if not isinstance(agent, HeteroRLAgent):
            raise ValueError(f"agent {agent_id} is not HeteroRLAgent.")
        if not isinstance(market, TotalTimeAwareMarket):
            raise ValueError(f"market is not TotalTimeAwareMarket.")
        remaining_time_ratio: float = self._calc_remaining_time_ratio(market)
        current_time: int = market.get_time()
        last_order_time: int = agent.last_order_time
        if current_time < last_order_time:
            raise ValueError(f"current_time {current_time} is less than last_order_time {last_order_time}.")
        market_prices: list[float] = market.get_market_prices(
            times=[t for t in range(last_order_time, current_time)]
        )
        if market.get_time() % 2000 == 1999:
            print(f"t={market.get_time()}, p_f={market.get_fundamental_price():.1f}, p_t={market.get_market_price():.1f}, asset_volume={agent.asset_volumes[market.market_id]}")
        log_return: float = self._calc_return(market_prices)
        volatility: float = self._calc_volatility(market_prices)
        self.return_dic[agent_id] = log_return
        self.volatility_dic[agent_id] = volatility
        asset_volume_buy_orders_ratio: float = self._get_asset_volume_existing_orders_ratio(
            agent, market, is_buy=True
        )
        asset_volume_sell_orders_ratio: float = self._get_asset_volume_existing_orders_ratio(
            agent, market, is_buy=False
        )
        blurred_fundamental_return: float = self._blur_fundamental_return(agent, market)
        skill_boundedness: float = agent.skill_boundedness
        if not hasattr(agent, "risk_aversion_term"):
            raise ValueError(f"agent {agent.agent_id} does not have risk_aversion_term.")
        risk_aversion_term: float = agent.risk_aversion_term
        if not hasattr(agent, "discount_factor"):
            raise ValueError(f"agent {agent.agent_id} does not have discount_factor.")
        discount_factor: float = agent.discount_factor
        obs: ObsType = np.array(
            [
                asset_ratio, liquidable_asset_ratio, 
                inverted_buying_power, remaining_time_ratio, log_return, volatility,
                asset_volume_buy_orders_ratio, asset_volume_sell_orders_ratio,
                blurred_fundamental_return, skill_boundedness, risk_aversion_term, discount_factor
            ]
        )
        obs = np.clip(obs, -3, 3)
        return obs
        
    def _calc_asset_ratio(
        self,
        agent: Agent,
        market: Market    
    ) -> float:
        """Calculate the percentage of the agent's total wealth accounted for by the asset."""
        market_price: float = market.get_market_price()
        cash_amount: float = agent.cash_amount
        asset_volume: float = agent.asset_volumes[market.market_id]
        asset_value: float = asset_volume * market_price
        total_wealth: float = cash_amount + asset_value
        asset_ratio: float = asset_value / (total_wealth + 1e-06)
        return asset_ratio
    
    def _liquidable_asset_ratio(
        self,
        agent: Agent,
        market: Market
    ) -> float:
        """Calculate holding asset volume / max order volume."""
        asset_volume: float = agent.asset_volumes[market.market_id]
        liquidable_asset_ratio: float = asset_volume / (self.max_order_volume + 1e-06)
        return liquidable_asset_ratio
    
    def _calc_inverted_buying_power(self, agent: Agent, market: Market) -> float:
        """Calculate market price / holding cash amount."""
        market_price: float = market.get_market_price()
        cash_amount: float = agent.cash_amount
        inverted_buying_power: float = market_price / (cash_amount + 1e-06)
        return inverted_buying_power
    
    def _calc_remaining_time_ratio(self, market: TotalTimeAwareMarket) -> float:
        """Calculate (T-t) / T where T is the total number of time steps in the episode."""
        total_time: Optional[int] = market.total_iteration_steps
        remaining_time: int = market.get_remaining_time()
        if total_time is None:
            return 1
        remaining_time_ratio: float = remaining_time / (total_time + 1e-06)
        return remaining_time_ratio
    
    def _calc_return(self, market_prices: list[float]) -> float:
        """Calculate return log p_t - log p_{t-tau} where tau is the time interval between the current and previous observations."""
        if len(market_prices) < 2:
            return 0.0
        log_return: float = np.log(market_prices[-1]) - np.log(market_prices[0])
        return log_return

    def _calc_volatility(self, market_prices: list[float]) -> float:
        """Calculate volatility between t-tau and t."""
        if len(market_prices) < 2:
            return 0.0
        log_returns: ndarray = np.diff(np.log(market_prices))
        volatility: float
        volatility = np.sum(log_returns ** 2) / len(log_returns)
        return volatility
    
    def _get_asset_volume_existing_orders_ratio(
        self, agent: Agent, market: Market, is_buy: bool
    ) -> float:
        """Calculate the percentage of the number of buy and sell limit orders with prices between [p_t-0.01p_t, p_t+0.01p_t] accounted for by the agent's holding asset volume."""
        price_volume_dic: dict[Optional[float], int] = market.get_buy_order_book() if is_buy \
            else market.get_sell_order_book()
        mid_price: float = market.get_mid_price()
        if mid_price is None:
            mid_price = market.get_market_price()
        lower_bound: float = mid_price - self.depth_range * mid_price
        upper_bound: float = mid_price + self.depth_range * mid_price
        existing_orders_volume: int = 0
        for price in price_volume_dic.keys():
            volume: int = price_volume_dic[price]
            if price is None:
                existing_orders_volume += volume
            elif (lower_bound <= price) and (price <= upper_bound):
                existing_orders_volume += volume
        asset_volume: int = np.abs(agent.asset_volumes[market.market_id])
        asset_volume_existing_orders_ratio: float = asset_volume / (existing_orders_volume + 1e-06)
        return asset_volume_existing_orders_ratio
    
    def _blur_fundamental_return(self, agent: Agent, market: Market) -> float:
        """Calculate log difference log p_t^f - log p_t blurred according to the agent's skill boundedness."""
        fundamental_price: float = market.get_fundamental_price()
        market_price: float = market.get_market_price()
        blurred_fundamental_return: float = np.log(fundamental_price) - np.log(market_price)
        if not hasattr(agent, "skill_boundedness"):
            raise ValueError(f"agent {agent.agent_id} does not have skill_boundedness.")
        skill_boundedness: float = agent.skill_boundedness
        blurring_factor: float = self._prng.gauss(0, skill_boundedness)
        blurred_fundamental_return += blurring_factor
        return blurred_fundamental_return

    def generate_reward(self, agent_id: AgentID) -> float:
        agent: HeteroRLAgent = self.simulator.agents[agent_id]
        cash_amount: float = agent.cash_amount
        if not hasattr(agent, "previous_utility"):
            raise ValueError(f"agent {agent_id} does not have previous_utility.")
        previous_utility: float = agent.previous_utility
        market: TotalTimeAwareMarket = self.simulator.markets[0]
        asset_volume: int = agent.asset_volumes[market.market_id]
        market_price: float = market.get_market_price()
        total_wealth: float = cash_amount + asset_volume * market_price
        log_return: float = self.return_dic[agent_id]
        volatility: float = self.volatility_dic[agent_id]
        current_utility: float = (
            total_wealth + asset_volume * market_price * log_return
        ) - 0.5 * agent.risk_aversion_term * (
            (asset_volume * market_price) ** 2
        ) * volatility
        if current_utility < 0:
            reward: float = self.negative_utility_penality
        elif previous_utility < 0:
            reward: float = 0.0
        else:
            reward: float = np.log(current_utility / previous_utility)
        if asset_volume < 0:
            reward: float = self.short_selling_penalty * np.abs(asset_volume)
        agent.previous_utility = current_utility
        return reward

    def generate_info(self, agent_id: AgentID) -> InfoType:
        pass

    def convert_action2orders(self, action: ActionType) -> list[Order | Cancel]:
        order_price_scale, order_volume_scale = action
        market: TotalTimeAwareMarket = self.simulator.markets[0]
        mid_price: float = market.get_mid_price()
        if mid_price is None:
            mid_price = market.get_market_price()
        is_buy: bool
        if 0 < order_volume_scale:
            order_price: float = mid_price - self.limit_order_range * mid_price * order_price_scale
            is_buy = True
        else:
            order_price: float = mid_price + self.limit_order_range * mid_price * order_price_scale
            is_buy = False
        order_volume: int = np.abs(
            np.ceil(self.max_order_volume * order_volume_scale)
        )
        if market.get_time() % 2000 == 1999:
            print(f"t={market.get_time()}, order_price={order_price:.1f}, order_volume={order_volume}")
        agent_id: AgentID = self.agent_selection
        if order_volume == 0:
            return []
        order: Order = Order(
            agent_id=agent_id,
            market_id=market.market_id,
            price=order_price,
            volume=order_volume,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            ttl=self.num_agents
        )
        return [order]
    
    def get_obs_names(self) -> list[str]:
        return [
            "asset_ratio", "inverted_buying_power", "remaining_time_ratio", "log_return", "volatility",
            "asset_volume_buy_orders_ratio", "asset_volume_sell_orders_ratio",
            "blurred_fundamental_return", "skill_boundedness", "risk_aversion_term", "discount_factor"
        ]
    
    def get_action_names(self) -> list[str]:
        return ["order_price_scale", "order_volume_scale"]
    
    def __str__(self) -> str:
        description: str = "[bold green]AECEnv4HeteroRL[/bold green]\n"
        description += f"max order volume: {self.max_order_volume} " + \
            f"limit order range: {self.limit_order_range} short selling penalty: {self.short_selling_penalty}\n"
        obs_names: list[str] = self.get_obs_names()
        description += f"obs: {obs_names}\n"
        action_names: list[str] = self.get_action_names()
        description += f"action: {action_names}"
        return description
        
