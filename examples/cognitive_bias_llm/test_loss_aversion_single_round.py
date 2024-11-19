import argparse
import csv
import json
import numpy as np
import random
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from envs.agents import fetch_llm_output

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--num_simulations", type=int, default=1)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--cash_amount", type=float, default=30000.0)
    parser.add_argument("--bought_price", type=float, default=300.0)
    parser.add_argument("--bought_volume", type=int, default=10)
    parser.add_argument("--is_uptrend", action="store_true")
    parser.add_argument("--is_peak", action="store_true")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument("--temp", type=float, default=0.7)
    return parser

premise: str = "You are a participant of the simulation of stock markets. " + \
    "Behave as an investor. Answer your order decision after analysing the given information. "
instruction: str = "\\n\\nYour current portfolio is provided as a following format. " + \
    "Unrealized gain refers to the increase in value of the investment that has not yet been sold. " + \
    "It represents the potential profit on your stock position. Negative unrealized gain means that " + \
    "the investment has decreased in value. " + \
    "\\n[Your portfolio]cash: {}\\n" + \
    "[Your portfolio]market id: {}, volume: {}, unrealized gain: {}, ..." + \
    "\\n\\nEach market condition is provided as a following format." + \
    "\\n[Market condition]market id: {}, current market price: {}, " + \
    "daily all time high price: {}, daily all time low price: {}, ..." + \
    "\\n\\nYour trading history is provided as a following format. " + \
    "Negative volume means that you sold the stock." + \
    "\\n[Your trading history]market id: {}, price: {}, volume: {}, ..."
answer_format: str = "\\n\\nDecide your investment in the following JSON format. " + \
    "Do not deviate from the format, " + \
    "and do not add any additional words to your response outside of the format. " + \
    "Make sure to enclose each property in double quotes. " + \
    "Order volume means the number of units you want to buy or sell the stock. " + \
    "Negative order volume means that you want to sell the stock. " + \
    "Order volume ranges from -10 to 10. " + \
    "Short selling is not allowed. Try to keep your order volume as non-zero and not-extreme as possible. " + \
    "Order price means the limit price at which you want to buy or sell the stock. By adjusting " + \
    "order price, you can trade at a more favorable price or adjust the time it takes to execute a trade. " + \
    "Here are the answer format." + \
    '\\n{<market id>: {order_price: <order price>, order_volume: <order volume>, reason: <reason>} ...}'

def create_info(
    cash_amount: float,
    bought_price: float,
    bought_volume: int,
    current_price: float,
    all_time_higth: float,
    all_time_low: float
) -> str:
    unrealized_gain: float = bought_volume * (current_price - bought_price)
    portfolio_info: str = f"\\n[Your portfolio]cash: {cash_amount}\\n" + \
        f"[Your portfolio]market id: 0, volume: {bought_volume}, unrealized gain: {unrealized_gain}"
    market_info: str = f"\\n[Market condition]market id: 0, current market price: {current_price}, " + \
        f"daily all time high price: {all_time_higth}, daily all time low price: {all_time_low}"
    trading_history: str = f"\\n[Your trading history]market id: 0, price: {bought_price} volume: {bought_volume}"
    return portfolio_info + market_info + trading_history

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    initial_seed: int = all_args.initial_seed
    num_simulations: int = all_args.num_simulations
    csv_path: Path = pathlib.Path(all_args.csv_path).resolve()
    cash_amount: float = all_args.cash_amount
    bought_price: float = all_args.bought_price
    bought_volume: int = all_args.bought_volume
    is_uptrend: bool = all_args.is_uptrend
    is_peak: bool = all_args.is_peak
    temp: float = all_args.temp
    llm_name: str = all_args.llm_name
    columns = [
        "cash_amount", "bought_price", "bought_volume", "current_price", "all_time_high", "all_time_low",
        "order_price", "order_volume", "reason"
    ]
    with open(csv_path, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(num_simulations):
            seed = initial_seed + i
            prng = random.Random(seed)
            log_return: float = prng.uniform(a=0, b=0.5)
            if not is_uptrend:
                log_return = -log_return
            current_price: float = bought_price * np.exp(log_return)
            if is_peak:
                if is_uptrend:
                    all_time_high: float = current_price
                    all_time_low: float = bought_price
                else:
                    all_time_low: float = current_price
                    all_time_high: float = bought_price
            else:
                if is_uptrend:
                    all_time_high: float = current_price * np.exp(log_return)
                    all_time_low: float = bought_price
                else:
                    all_time_low: float = current_price * np.exp(log_return)
                    all_time_high: float = bought_price
            info: str = create_info(
                cash_amount, bought_price, bought_volume, current_price, all_time_high, all_time_low
            )
            prompt: str = premise + f"\\n\\nHere's the information." + instruction + info + answer_format
            prompt = json.dumps(
                {"text": prompt, "temperature": temp},
                ensure_ascii=False
            )
            llm_output: str = fetch_llm_output(prompt, llm_name)
            if llm_output[:7] == "```json" and llm_output[-3:] == "```":
                llm_output = llm_output[7:-3]
            try:
                order_dic: dict[int, int] = json.loads(llm_output)["0"]
            except Exception as e:
                print(e)
                continue
            if not "order_price" in order_dic:
                raise ValueError("order_price not found.")
            else:
                order_price: float = float(order_dic["order_price"])
            if not "order_volume" in order_dic:
                raise ValueError("order_volume not found.")
            else:
                order_volume: int = int(order_dic["order_volume"])
            if not "reason" in order_dic:
                raise ValueError("reason not found.")
            else:
                reason: str = order_dic["reason"]
            writer.writerow(
                [cash_amount, bought_price, bought_volume, current_price,
                all_time_high, all_time_low, order_price, order_volume, reason]
            )
        
if __name__ == "__main__":
    main(sys.argv[1:])
        



