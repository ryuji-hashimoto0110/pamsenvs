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
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeVar
import subprocess
from rich import print

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

def prepare_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    special_tokens: list[str] = ["<eos>", "<unk>", "<sep>", "<pad>", "<cls>", "<mask>"]
    for token in special_tokens:
        if tokenizer.convert_tokens_to_ids(token) is None:
            tokenizer.add_tokens([token])
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")
    return tokenizer

def fetch_llm_output(
    prompt: str,
    llm_name: Literal[
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    device: torch.device,
    model: Optional[PreTrainedModel] = None,
) -> tuple[str, PreTrainedModel]:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer = prepare_tokenizer(tokenizer)
    if model is None:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float16, #pad_token_id=tokenizer.eos_token_id
        )
    model.to(device)
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs: dict[str, Any] = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs: dict[str, Any] = model.generate(
        **inputs, pad_token_id=tokenizer.eos_token_id,
        max_length=1024, do_sample=True, temperature=0.7
    )
    llm_output: str = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).split("assistant")[-1].strip()
    return llm_output, model

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
        self.executed_orders_dic: dict[MarketID, list[ExecutionLog]] = {}
        for market_id in accessible_markets_ids:
            self.executed_orders_dic[market_id] = []

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
        self,
        markets: list[Market],
    ) -> list[Order | Cancel]:
        """submit orders.
        """
        prompt: str = self.create_prompt(markets=markets)
        success: bool = False
        llm_output: str = fetch_llm_output(
            prompt=prompt, llm_name=self.llm_name
        )
        if llm_output[:7] == "```json" and llm_output[-3:] == "```":
            llm_output = llm_output[7:-3]
        exo_order_price_dic: Optional[dict[MarketID, float]]
        exo_order_volume_dic: Optional[dict[MarketID, int]]
        exo_order_price_dic, exo_order_volume_dic = self.get_exo_order_prices_volumes(
            markets=markets
        )
        orders: list[Order | Cancel] = self.convert_llm_output2orders(
            llm_output=llm_output, markets=markets,
            exo_order_prices=exo_order_price_dic,
            exo_order_volumes=exo_order_volume_dic
        )
        return orders
    
    def executed_order(self, log: ExecutionLog) -> None:
        market_id: MarketID = log.market_id
        self.executed_orders_dic[market_id].append(log)

    def get_exo_order_prices_volumes(
        self,
        markets: list[Market]
    ) -> tuple[Optional[list[float]], Optional[list[int]]]:
        """get exogenous order prices and volumes.
        
        Args:
            markets (list[Market]): list of markets.
        
        Returns:
            tuple[list[float], list[int]]: exogenous order prices and volumes.
        """
        return None, None
