from abc import abstractmethod
import json
import math
from pams.agents import Agent
from pams.agents import HighFrequencyAgent
from pams.logs.base import ExecutionLog
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeVar
import subprocess
from rich import print

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

def fetch_llm_output(
    prompt: str,
    llm_name: Literal["gpt-4o-mini", "gpt-4o", "llama", "finllama"]
) -> str:
    commands: list[str] = [
        'curl', '-X', 'POST', '-H',
        'Content-Type: application/json', '-d', 
        f'{{\"text\":\"{prompt}\"}}',
        f'http://hpc15.socsim.t.u-tokyo.ac.jp:8000/{llm_name}'
    ]
    raw_llm_output: str = subprocess.run(
        commands, capture_output=True, text=True
    ).stdout
    llm_output_dic: dict[str, str] = json.loads(raw_llm_output)
    llm_output: str = llm_output_dic["response"]
    return llm_output

class PromptAwareAgent(Agent):
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
            and can include the parameter:
                base_prompt (str): the base prompt for the agent.
            accessible_markets_ids (list[MarketID]): list of accessible market ids.
        """
        super().setup(
            settings=settings, accessible_markets_ids=accessible_markets_ids
        )
        if "basePrompt" in settings:
            self.base_prompt: Optional[str] = settings["basePrompt"]
        else:
            self.base_prompt: Optional[str] = None
        self.llm_name: str = settings["llmName"]
        self.executed_orders_dic: dict[MarketID, list[float]] = {}

    @abstractmethod
    def create_prompt(self, markets: list[Market]) -> str:
        """create a prompt for the agent."""
        pass

    @abstractmethod
    def convert_llm_output2orders(
        self,
        llm_output: str,
        markets: list[Market]
    ) -> list[Order | Cancel]:
        """convert the LLM output to orders."""
        pass

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order | Cancel]:
        """submit orders.
        """
        prompt: str = self.create_prompt(markets=markets)
        success: bool = False
        for _ in range(10):
            try:
                llm_output: str = fetch_llm_output(
                    prompt=prompt, llm_name=self.llm_name
                )
                if llm_output[:7] == "```json" and llm_output[-3:] == "```":
                    llm_output = llm_output[7:-3]
                print("[green]==llm output==[green]")
                print(llm_output)
                orders: list[Order | Cancel] = self.convert_llm_output2orders(
                    llm_output=llm_output, markets=markets
                )
                success = True
            except Exception:
                continue
            if success:
                break
        if not success:
            raise ValueError(f"Failed to convert the LLM output to orders: {llm_output}.")
        return orders
    
    def executed_order(self, log: ExecutionLog) -> None:
        market_id: MarketID = log.market_id
        if market_id not in self.executed_orders_dic:
            self.executed_orders_dic[market_id] = [log]
        else:
            self.executed_orders_dic[market_id].append(log)
