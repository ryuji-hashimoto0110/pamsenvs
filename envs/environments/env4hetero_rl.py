from .environment import PamsAECEnv
from gymnasium import spaces
from gymnasium import Space
from pams.agents import Agent
from pams.logs import Logger
from pams.order import Cancel
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
        self.variables_dic: dict[str, dict[str, float]] = {}
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
            episode_config_dic=episode_config_dic,
            logger=logger,
            prng=self._prng
        )
        self._register_classes(runner)
        runner._setup()
        return runner

    def _register_classes(self, runner: Runner) -> None:
        pass

    def _get_agent_traits_dic(
        self,
        runner: Runner
    ) -> dict[AgentID, dict[str, float | dict[MarketID, float]]]:
        """get agent traits.

        'agent traits' are the attributes of agents that are used to generate 
        observations and rewards, containing:
            - last order time
            - holding asset volume
            - holding cash amount
            - skill boundedness
            - risk aversion term
            - discount factor
        """
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

    def add_attributes(self) -> None:
        self.agent_traits_dic: dict[AgentID, dict[str, float]] = self._get_agent_traits_dic(self.runner)

    def generate_obs(self, agent_id: AgentID) -> ObsType:
        pass

    def generate_reward(self, agent_id: AgentID) -> float:
        pass

    def generate_info(self, agent_id: AgentID) -> InfoType:
        pass

    def convert_action2orders(self, action: ActionType) -> list[Order | Cancel]:
        pass
