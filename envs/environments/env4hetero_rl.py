from ..agents import CARAFCNAgent
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
import scipy.stats as stats
from typing import Literal
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
        obs_names: list[str] = [],
        action_names: list[str] = [],
        depth_range: float = 0.01,
        limit_order_range: float = 0.1,
        max_order_volume: int = 10,
        short_selling_penalty: float = 0.5,
        cash_shortage_penalty: float = 0.5,
        execution_vonus: float = 0.1,
        execution_vonus_decay: float = 0.9,
        initial_fundamental_penalty: float = 1.0,
        fundamental_penalty_decay: float = 0.9,
        agent_trait_memory: float = 0.9,
        store_experience_time: int = 300
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
        assert len(obs_names) == obs_dim, f"length of the obs_names {len(obs_names)} is not equal to obs_dim {obs_dim}."
        self.obs_names = obs_names
        assert len(action_names) == action_dim, f"length of the action_names {len(action_names)} is not equal to action_dim {action_dim}."
        self.action_names = action_names
        self.variables_dic: dict[str, dict[str, float]] = {}
        self.depth_range: float = depth_range
        self.limit_order_range: float = limit_order_range
        self.max_order_volume: int = max_order_volume
        self.short_selling_penalty: float = short_selling_penalty
        self.cash_shortage_penalty: float = cash_shortage_penalty
        self.execution_vonus: float = execution_vonus
        self.execution_vonus_decay: float = execution_vonus_decay
        self.fundamental_penalty: float = initial_fundamental_penalty
        self.fundamental_penalty_decay: float = fundamental_penalty_decay
        self.agent_trait_memory: float = agent_trait_memory
        self.previous_agent_trait_dic: dict[AgentID, dict[str, float]] = {}
        self.store_experience_time: int = store_experience_time

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
        runner.class_register(CARAFCNAgent)
        runner.class_register(HeteroRLAgent)
        runner.class_register(DividendProviderwEverySteps)

    def reset(self) -> None:
        super().reset()
        self.execution_vonus *= self.execution_vonus_decay
        self.fundamental_penalty *= self.fundamental_penalty_decay

    def is_ready_to_store_experience(self) -> bool:
        if self.get_time() < self.store_experience_time:
            return False
        else:
            return True

    def add_attributes(self) -> None:
        self.num_execution_dic: dict[AgentID, int] = {}
        self.return_dic: dict[AgentID, float] = {}
        self.volatility_dic: dict[AgentID, float] = {}
        for agent_id in self.agents:
            self.return_dic[agent_id] = 0.0
            self.volatility_dic[agent_id] = 0.0
            self.num_execution_dic[agent_id] = 0
            self._smooth_agent_trait(agent_id)
        self.obs_dic: dict[str, list[float | int]] = {
            "step": [], "agent_id": [], "asset_volume": [], "cash_amount": [],
            "market_price": [], "fundamental_price": []
        }
        for obs_name in self.obs_names:
            self.obs_dic[obs_name] = []
        self.action_dic: dict[str, list[float | int]] = {
            "step": [], "agent_id": [], "order_price": [], "order_volume": [], "is_buy": []
        }
        for action_name in self.action_names:
            self.action_dic[action_name] = []
        self.reward_dic: dict[str, list[float | int]] = {
            "step": [], "agent_id": [], "scaled_utility_diff": [],
            "short_selling_penalty": [], "cash_shortage_penalty": [],
            "fundamental_penalty": [],
            "execution_vonus": [], "total_reward": []
        }

    def _smooth_agent_trait(self, agent_id: AgentID) -> None:
        agent: HeteroRLAgent = self.simulator.agents[agent_id]
        if agent_id in self.previous_agent_trait_dic.keys():
            previous_skill_boundedness: float = self.previous_agent_trait_dic[agent_id]["skill_boundedness"]
            new_skill_boundedness: float = self.agent_trait_memory * previous_skill_boundedness + \
                (1 - self.agent_trait_memory) * agent.skill_boundedness
            agent.skill_boundedness = new_skill_boundedness
            previous_risk_aversion_term: float = self.previous_agent_trait_dic[agent_id]["risk_aversion_term"]
            new_risk_aversion_term: float = self.agent_trait_memory * previous_risk_aversion_term + \
                (1 - self.agent_trait_memory) * agent.risk_aversion_term
            agent.risk_aversion_term = new_risk_aversion_term
            previous_discount_factor: float = self.previous_agent_trait_dic[agent_id]["discount_factor"]
            new_discount_factor: float = min(
                1,
                self.agent_trait_memory * previous_discount_factor + \
                    (1 - self.agent_trait_memory) * agent.discount_factor
            )
            agent.discount_factor = new_discount_factor
        self.previous_agent_trait_dic[agent_id] = {
            "skill_boundedness": agent.skill_boundedness,
            "risk_aversion_term": agent.risk_aversion_term,
            "discount_factor": agent.discount_factor
        }

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
        obs_list: list[float] = []
        agent: HeteroRLAgent = self.simulator.agents[agent_id]
        market: TotalTimeAwareMarket = self.simulator.markets[0]
        current_time: int = market.get_time()
        self.obs_dic["step"].append(current_time)
        self.obs_dic["agent_id"].append(agent_id)
        self.obs_dic["asset_volume"].append(agent.asset_volumes[market.market_id])
        self.obs_dic["cash_amount"].append(agent.cash_amount)
        self.obs_dic["market_price"].append(market.get_market_price())
        self.obs_dic["fundamental_price"].append(market.get_fundamental_price())
        asset_ratio: float = self._calc_asset_ratio(agent, market)
        if "asset_ratio" in self.obs_names:
            asset_ratio = self._preprocess_obs(asset_ratio, "asset_ratio")
            obs_list.append(asset_ratio)
            self.obs_dic["asset_ratio"].append(asset_ratio)
        liquidable_asset_ratio: float = self._liquidable_asset_ratio(agent, market)
        if "liquidable_asset_ratio" in self.obs_names:
            liquidable_asset_ratio = self._preprocess_obs(liquidable_asset_ratio, "liquidable_asset_ratio")
            obs_list.append(liquidable_asset_ratio)
            self.obs_dic["liquidable_asset_ratio"].append(liquidable_asset_ratio)
        inverted_buying_power: float = self._calc_inverted_buying_power(agent, market)
        if "inverted_buying_power" in self.obs_names:
            inverted_buying_power = self._preprocess_obs(inverted_buying_power, "inverted_buying_power")
            obs_list.append(inverted_buying_power)
            self.obs_dic["inverted_buying_power"].append(inverted_buying_power)
        if not isinstance(agent, HeteroRLAgent):
            raise ValueError(f"agent {agent_id} is not HeteroRLAgent.")
        if not isinstance(market, TotalTimeAwareMarket):
            raise ValueError(f"market is not TotalTimeAwareMarket.")
        remaining_time_ratio: float = self._calc_remaining_time_ratio(market)
        if "remaining_time_ratio" in self.obs_names:
            remaining_time_ratio = self._preprocess_obs(remaining_time_ratio, "remaining_time_ratio")
            obs_list.append(remaining_time_ratio)
            self.obs_dic["remaining_time_ratio"].append(remaining_time_ratio)
        last_order_time: int = agent.last_order_time
        if current_time < last_order_time:
            raise ValueError(f"current_time {current_time} is less than last_order_time {last_order_time}.")
        market_prices: list[float] = market.get_market_prices(
            times=[t for t in range(last_order_time, current_time)]
        )
        log_return: float = self._calc_return(market_prices)
        self.return_dic[agent_id] = log_return
        if "log_return" in self.obs_names:
            log_return = self._preprocess_obs(log_return, "log_return")
            obs_list.append(log_return)
            self.obs_dic["log_return"].append(log_return)
        volatility: float = self._calc_volatility(market_prices)
        self.volatility_dic[agent_id] = volatility
        if "volatility" in self.obs_names:
            volatility = self._preprocess_obs(volatility, "volatility")
            obs_list.append(volatility)
            self.obs_dic["volatility"].append(volatility)
        asset_volume_buy_orders_ratio: float = self._get_asset_volume_existing_orders_ratio(
            agent, market, is_buy=True
        )
        if "asset_volume_buy_orders_ratio" in self.obs_names:
            asset_volume_buy_orders_ratio = self._preprocess_obs(
                asset_volume_buy_orders_ratio, "asset_volume_buy_orders_ratio"
            )
            obs_list.append(asset_volume_buy_orders_ratio)
            self.obs_dic["asset_volume_buy_orders_ratio"].append(asset_volume_buy_orders_ratio)
        asset_volume_sell_orders_ratio: float = self._get_asset_volume_existing_orders_ratio(
            agent, market, is_buy=False
        )
        if "asset_volume_sell_orders_ratio" in self.obs_names:
            asset_volume_sell_orders_ratio = self._preprocess_obs(
                asset_volume_sell_orders_ratio, "asset_volume_sell_orders_ratio"
            )
            obs_list.append(asset_volume_sell_orders_ratio)
            self.obs_dic["asset_volume_sell_orders_ratio"].append(asset_volume_sell_orders_ratio)
        blurred_fundamental_return: float = self._blur_fundamental_return(agent, market)
        if "blurred_fundamental_return" in self.obs_names:
            blurred_fundamental_return = self._preprocess_obs(
                blurred_fundamental_return, "blurred_fundamental_return"
            )
            obs_list.append(blurred_fundamental_return)
            self.obs_dic["blurred_fundamental_return"].append(blurred_fundamental_return)
        skill_boundedness: float = agent.skill_boundedness
        if "skill_boundedness" in self.obs_names:
            skill_boundedness = self._preprocess_obs(skill_boundedness, "skill_boundedness")
            obs_list.append(skill_boundedness)
            self.obs_dic["skill_boundedness"].append(skill_boundedness)
        if not hasattr(agent, "risk_aversion_term"):
            raise ValueError(f"agent {agent.agent_id} does not have risk_aversion_term.")
        risk_aversion_term: float = agent.risk_aversion_term
        if "risk_aversion_term" in self.obs_names:
            risk_aversion_term = self._preprocess_obs(risk_aversion_term, "risk_aversion_term")
            obs_list.append(risk_aversion_term)
            self.obs_dic["risk_aversion_term"].append(risk_aversion_term)
        if not hasattr(agent, "discount_factor"):
            raise ValueError(f"agent {agent.agent_id} does not have discount_factor.")
        discount_factor: float = agent.discount_factor
        if "discount_factor" in self.obs_names:
            discount_factor = self._preprocess_obs(discount_factor, "discount_factor")
            obs_list.append(discount_factor)
            self.obs_dic["discount_factor"].append(discount_factor)
        obs: ObsType = np.array(obs_list)
        return obs
    
    def _preprocess_obs(
        self,
        obs_comp: float,
        obs_name: Literal[
            "asset_ratio", "liquidable_asset_ratio", "inverted_buying_power",
            "remaining_time_ratio", "log_return", "volatility",
            "asset_volume_buy_orders_ratio", "asset_volume_sell_orders_ratio",
            "blurred_fundamental_return", "skill_boundedness", "risk_aversion_term",
            "discount_factor"
        ]
    ) -> float:
        if obs_name == "asset_ratio":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 1)
        elif obs_name == "liquidable_asset_ratio":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 3)
        elif obs_name == "inverted_buying_power":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 1)
        elif obs_name == "remaining_time_ratio":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 1)
        elif obs_name == "log_return":
            obs_comp = self._minmax_rescaling(obs_comp, -0.1, 0.1)
        elif obs_name == "volatility":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 0.03)
        elif obs_name == "asset_volume_buy_orders_ratio":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 2)
        elif obs_name == "asset_volume_sell_orders_ratio":
            obs_comp = self._minmax_rescaling(obs_comp, 0, 2)
        elif obs_name == "blurred_fundamental_return":
            obs_comp = self._minmax_rescaling(obs_comp, -0.1, 0.1)
        elif obs_name == "skill_boundedness":
            obs_comp = self._minmax_rescaling(
                obs_comp,
                setting=self.config_dic["Agent"]["skillBoundedness"]
            )
        elif obs_name == "risk_aversion_term":
            obs_comp = self._minmax_rescaling(
                obs_comp,
                setting=self.config_dic["Agent"]["riskAversionTerm"]
            )
        elif obs_name == "discount_factor":
            obs_comp = self._minmax_rescaling(
                obs_comp,
                max_val=1,
                setting=self.config_dic["Agent"]["discountFactor"]
            )
        else:
            raise NotImplementedError(f"obs_name {obs_name} is not implemented.")
        obs_comp = np.clip(obs_comp, -1, 1)
        return obs_comp

    def _minmax_rescaling(
        self,
        x: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        setting: Optional[float | list[float] | dict[str, float | list[float]]] = None,
    ) -> float:
        """Min-max rescaling. set the variable x to the range [-1, 1]."""
        if max_val is not None and min_val is not None:
            rescaled_x: float = 2 * (x - min_val) / (max_val - min_val) - 1
        elif setting is not None:
            if min_val is None:
                min_val = self._get_percentile(setting, 0.999)
            if max_val is None:
                max_val = self._get_percentile(setting, 0.001)
            if min_val == max_val:
                rescaled_x: float = 0
            else:
                rescaled_x: float = 2 * (x - min_val) / (max_val - min_val) - 1
        else:
            raise ValueError("Either min_val and max_val or setting must be specified.")
        return rescaled_x
    
    def _get_percentile(
        self,
        setting: float | list[float] | dict[str, float | list[float]],
        upper_prob: float = 0.01
    ) -> float:
        """get percentile.
        
        This method is usually used to get the maximum value of the distribution of any variable.
        
        Args:
            setting (float | list[float] | dict[str, float | list[float]]): setting of the distribution.
                Ex: 1.0, [0, 10], {"expon": [1.0]}, {"uniform": [0, 10]}
            upper_prob (float): upper probability. Defaults to 0.01.
        """
        if isinstance(setting, float):
            return setting
        elif isinstance(setting, list):
            assert len(setting) == 2
            umin, umax = setting
            return umax - (umax - umin) * upper_prob
        elif isinstance(setting, dict):
            if "expon" in setting:
                lam: float = setting["expon"][0]
                return - lam * np.log(upper_prob)
            elif "uniform" in setting:
                assert len(setting["uniform"]) == 2
                umin, umax = setting["uniform"]
                return umax - (umax - umin) * upper_prob
            elif "normal" in setting:
                assert len(setting["normal"]) == 2
                mu, sigma = setting["normal"]
                return stats.norm.ppf(
                    1-upper_prob, loc=mu, scale=sigma
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
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
        #market_price_arr: ndarray = np.array(market_prices)
        log_return: float = np.log(market_prices[-1]) - np.log(market_prices[0])
        return log_return

    def _calc_volatility(self, market_prices: list[float]) -> float:
        """Calculate volatility between t-tau and t."""
        if len(market_prices) < 2:
            return 0.0
        log_return_arr: ndarray = np.log(market_prices[1:]) - np.log(market_prices[:len(market_prices)-1])
        avg_log_return: float = np.mean(log_return_arr)
        volatility: float = np.sum((log_return_arr - avg_log_return)**2)
        return volatility
    
    def _get_asset_volume_existing_orders_ratio(
        self, agent: Agent, market: Market, is_buy: bool
    ) -> float:
        """Calculate the percentage of the number of buy and sell limit orders with prices between [p_t-0.01p_t, p_t+0.01p_t] accounted for by the agent's holding asset volume.
        
        """
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
                weight: float = 1 / (1 + np.abs(price - mid_price) / mid_price)
                existing_orders_volume += volume * weight
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
        skill_boundedness: float = max(1e-10, agent.skill_boundedness)
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
        current_time: int = market.get_time()
        self.reward_dic["step"].append(current_time-1)
        self.reward_dic["agent_id"].append(agent_id)
        asset_volume: int = agent.asset_volumes[market.market_id]
        market_price: float = market.get_market_price()
        total_wealth: float = cash_amount + asset_volume * market_price
        log_return: float = self.return_dic[agent_id]
        volatility: float = self.volatility_dic[agent_id]
        asset_fraction: float = asset_volume * market_price / total_wealth
        risk_aversion_term: float = max(0, agent.risk_aversion_term)
        current_utility: float = (
            total_wealth * (1 + asset_fraction * log_return)
        ) - 0.5 * risk_aversion_term * asset_fraction * volatility * total_wealth
        agent.previous_utility = current_utility
        utility_diff = current_utility - previous_utility
        normalization_factor = max(abs(previous_utility), 1.0)
        scaled_utility_diff: float = utility_diff / normalization_factor
        #if scaled_utility_diff > 5:
        #    print(f"{previous_utility=:.2f}, {current_utility=:.2f} {scaled_utility_diff=:.2f}")
        #    print(f"{market.get_time()} {cash_amount=:.1f} {asset_volume=:.1f} {market_price=:.1f} {total_wealth=:.1f} {log_return=:.4f} {volatility=:.6f} alpha={agent.risk_aversion_term:.2f}")
        reward = scaled_utility_diff
        self.reward_dic["scaled_utility_diff"].append(scaled_utility_diff)
        if asset_volume < 0:
            short_selling_penalty: float = self.short_selling_penalty
            reward -= short_selling_penalty
        else:
            short_selling_penalty: float = 0
        self.reward_dic["short_selling_penalty"].append(-short_selling_penalty)
        if cash_amount < 0:
            cash_shortage_penalty: float = self.cash_shortage_penalty
            reward -= cash_shortage_penalty
        else:
            cash_shortage_penalty: float = 0
        self.reward_dic["cash_shortage_penalty"].append(-cash_shortage_penalty)
        execution_vonus: float = self.execution_vonus * agent.num_executed_orders
        reward += execution_vonus
        self.reward_dic["execution_vonus"].append(execution_vonus)
        fundamental_price: float = market.get_fundamental_price()
        fundamental_return: float = np.abs(
            np.log(fundamental_price) - np.log(market_price)
        )
        fundamental_penalty: float = self.fundamental_penalty * fundamental_return
        reward -= fundamental_penalty
        self.reward_dic["fundamental_penalty"].append(-fundamental_penalty)
        # print(f"{reward=:.2f}")
        # print(
        #     f"utility diff: {scaled_utility_diff:.3f} " + \
        #     f"short selling penalty: {self.reward_dic['short_selling_penalty'][-1]:.3f} " + \
        #     f"cash shortage penalty: {self.reward_dic['cash_shortage_penalty'][-1]:.3f} " + \
        #     f"fundamental penalty: {self.reward_dic['fundamental_penalty'][-1]:.3f} " + \
        #     f"execution vonus: {self.reward_dic['execution_vonus'][-1]:.3f}"
        # )
        self.reward_dic["total_reward"].append(reward)
        return reward

    def generate_info(self, agent_id: AgentID) -> InfoType:
        agent: HeteroRLAgent = self.simulator.agents[agent_id]
        self.num_execution_dic[agent_id] += agent.num_executed_orders
        return {"execution_volume": self.num_execution_dic[agent_id]}
        
    def convert_action2orders(self, action: ActionType) -> list[Order | Cancel]:
        order_price_scale, order_volume_scale = action
        self.action_dic["order_price_scale"].append(order_price_scale)
        self.action_dic["order_volume_scale"].append(order_volume_scale)
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
        self.action_dic["is_buy"].append(int(is_buy))
        order_volume: int = np.abs(
            np.ceil(self.max_order_volume * order_volume_scale)
        )
        agent_id: AgentID = self.agent_selection
        current_time: int = market.get_time()
        self.action_dic["step"].append(current_time)
        self.action_dic["agent_id"].append(agent_id)
        if order_volume == 0:
            self.action_dic["order_price"].append(None)
            self.action_dic["order_volume"].append(order_volume)
            return []
        order: Order = Order(
            agent_id=agent_id,
            market_id=market.market_id,
            price=order_price,
            volume=order_volume,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            ttl=len(self.simulator.agents)
        )
        self.action_dic["order_price"].append(order_price)
        self.action_dic["order_volume"].append(order_volume)
        return [order]
    
    def __str__(self) -> str:
        description: str = "[bold green]AECEnv4HeteroRL[/bold green]\n"
        description += f"max order volume: {self.max_order_volume} " + \
            f"limit order range: {self.limit_order_range} short selling penalty: {self.short_selling_penalty} " + \
            f"cash shortage penalty: {self.cash_shortage_penalty} " + \
            f"execution vonus: {self.execution_vonus} ({self.execution_vonus_decay})" + \
            f"fundamental penalty: {self.fundamental_penalty} ({self.fundamental_penalty_decay})\n"
        description += f"obs: {self.obs_names}\n"
        description += f"action: {self.action_names}"
        return description
        
