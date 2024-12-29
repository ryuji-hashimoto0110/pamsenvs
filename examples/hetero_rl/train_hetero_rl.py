import argparse
from argparse import ArgumentParser
import copy
import json
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from drl_algos import Algorithm
from drl_algos import IPPO
from drl_algos import Trainer
from envs.environments import AECEnv4HeteroRL
from pams.simulator import Simulator
import random
import torch
from typing import Any
from typing import Optional

def get_config() -> ArgumentParser:
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_name", type=str, default="ippo", choices=["ippo"]
    )
    parser.add_argument("--rollout_length", type=int, default=64)
    parser.add_argument("--num_updates_per_rollout", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_actor", type=float, default=5e-05)
    parser.add_argument("--lr_critic", type=float, default=1e-04)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--lmd", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_save_path", type=str)
    parser.add_argument("--actor_best_save_name", type=str)
    parser.add_argument("--actor_last_save_name", type=str)
    parser.add_argument("--num_train_steps", type=int, default=int(1e+08))
    parser.add_argument("--eval_interval", type=int, default=int(1e+05))
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--agent_name", type=str, default="heteroRLAgent")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--variable_ranges_path", type=str, required=False)
    parser.add_argument("--obs_names", type=str, nargs="+")
    parser.add_argument("--action_names", type=str, nargs="+")
    parser.add_argument("--depth_range", type=float, default=0.01)
    parser.add_argument("--limit_order_range", type=float, default=0.05)
    parser.add_argument("--max_order_volume", type=int, default=50)
    parser.add_argument("--short_selling_penalty", type=float, default=0.5)
    parser.add_argument("--execution_vonus", type=float, default=0.1)
    parser.add_argument("--agent_trait_memory", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.00, 0.03, 0.06, 0.09, 0.20]
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.00, 0.30, 0.60, 0.90, 2.00]
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95, 0.999]
    )
    return parser

def convert_str2path(
    path_name: Optional[str | Path],
    mkdir: bool
) -> Optional[Path]:
    """convert str to path.
    
    Args:
        path_name (str | Path, optional): Name of path. If path_name is str, it will be
            converted to Path. path_name is recommended to be an absolute path.
        mkdir (bool): Whether to make directory if path does not exist.
    """
    if path_name is None:
        return None
    elif isinstance(path_name, Path):
        path: Path = path_name.resolve()
    elif isinstance(path_name, str):
        path: Path = pathlib.Path(path_name).resolve()
    else:
        raise ValueError(
            f"path_name must be either str or pathlib.Path. path_name={path_name}"
        )
    if not path.exists():
        if mkdir:
            path.mkdir(parents=True)
        else:
            raise FileNotFoundError(f"path does not exist. path={path}")
    return path

def get_target_agent_names(config_dic: dict[str, Any], agent_name: str) -> list[str]:
    if agent_name not in config_dic:
        raise ValueError(f"{agent_name} is not in config_dic")
    agent_settings: dict[str, Any] = config_dic[agent_name]
    num_agents: int = 1
    id_from: int = 0
    id_to: int = 0
    if "numAgents" in agent_settings:
        num_agents = int(agent_settings["numAgents"])
        id_to = num_agents - 1
    if "from" in agent_settings or "to" in agent_settings:
        if "from" not in agent_settings or "to" not in agent_settings:
            raise ValueError(
                f"both {agent_name}.from and {agent_name}.to are required in json file if you use"
            )
        if "numAgents" in agent_settings:
            raise ValueError(
                f"{agent_name}.numAgents and ({agent_name}.from or {agent_name}.to) cannot be used at the same time"
            )
        id_from = int(agent_settings["from"])
        id_to = int(agent_settings["to"])
    prefix: str
    if "prefix" in agent_settings:
        prefix = agent_settings["prefix"]
    else:
        prefix = agent_name + "-"
    target_agent_names: list[str] = [
        prefix + str(i) for i in range(id_from, id_to+1)
    ]
    num_agents = len(target_agent_names)
    return target_agent_names, num_agents

def create_env(all_args, config_dic: dict[str, Any]) -> tuple[AECEnv4HeteroRL, int]:
    target_agent_names, num_agents = get_target_agent_names(
        config_dic=config_dic, agent_name=all_args.agent_name
    )
    variable_ranges_path: Path = convert_str2path(all_args.variable_ranges_path, mkdir=False)
    variable_ranges_dic: dict[str, list[float]] = json.load(
        fp=open(str(variable_ranges_path), mode="r")
    )
    obs_names: list[str] = all_args.obs_names
    action_names: list[str] = all_args.action_names
    env: AECEnv4HeteroRL = AECEnv4HeteroRL(
        config_dic=config_dic, variable_ranges_dic=variable_ranges_dic,
        simulator_class=Simulator, target_agent_names=target_agent_names,
        action_dim=len(action_names), obs_dim=len(obs_names), depth_range=all_args.depth_range,
        obs_names=obs_names, action_names=action_names,
        limit_order_range=all_args.limit_order_range,
        max_order_volume=all_args.max_order_volume,
        short_selling_penalty=all_args.short_selling_penalty,
        execution_vonus=all_args.execution_vonus,
        agent_trait_memory=all_args.agent_trait_memory,
    )
    return env, num_agents

def create_ippo(all_args, num_agents) -> IPPO:
    ippo: IPPO = IPPO(
        device=all_args.device,
        obs_shape=(12,), action_shape=(2,), num_agents=num_agents,
        seed=all_args.seed, rollout_length=all_args.rollout_length,
        num_updates_per_rollout=all_args.num_updates_per_rollout,
        batch_size=all_args.batch_size, gamma_idx=-1,
        lr_actor=all_args.lr_actor, lr_critic=all_args.lr_critic,
        clip_eps=all_args.clip_eps, lmd=all_args.lmd,
        max_grad_norm=all_args.max_grad_norm
    )
    return ippo

def main(args) -> None:
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    sigmas: list[float] = all_args.sigmas
    alphas: list[float] = all_args.alphas
    gammas: list[float] = all_args.gammas
    config_path: Path = convert_str2path(all_args.config_path, mkdir=False)
    config_dic: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    for sigma in sigmas:
        sigma_str: str = f"{sigma:.2f}".replace(".", "")
        config_dic["Agent"]["skillBoundedness"] = {"expon": [sigma]}
        for alpha in alphas:
            alpha_str: str = f"{alpha:.2f}".replace(".", "")
            config_dic["Agent"]["riskAversionTerm"] = {"expon": [alpha]}
            for gamma in gammas:
                gamma_str: str = f"{gamma:.2f}".replace(".", "")
                config_dic["Agent"]["discountFactor"] = {"uniform": [gamma, 0.999]}
                print(f"agent config: {config_dic['Agent']}")
                actor_save_path: Path = convert_str2path(all_args.actor_save_path, mkdir=True)
                actor_best_save_name = all_args.actor_best_save_name + f"-{sigma_str}-{alpha_str}-{gamma_str}-{all_args.seed}.pth"
                actor_last_save_name = all_args.actor_last_save_name + f"-{sigma_str}-{alpha_str}-{gamma_str}-{all_args.seed}.pth"
                actor_best_save_path: Path = actor_save_path / actor_best_save_name
                actor_last_save_path: Path = actor_save_path / actor_last_save_name
                train_env, num_agents = create_env(all_args, config_dic)
                test_env: AECEnv4HeteroRL = copy.deepcopy(train_env)
                test_env.fundamental_penalty = 0.0
                test_env.agent_trait_memory = 0.0
                ippo: IPPO = create_ippo(all_args, num_agents)
                trainer: Trainer = Trainer(
                    train_env=train_env, test_env=test_env, algo=ippo,
                    seed=all_args.seed,
                    actor_best_save_path=actor_best_save_path,
                    actor_last_save_path=actor_last_save_path,
                    other_indicators=[
                        "execution_volume", "lr_actor", "price_range",
                        "obs_dics", "action_dics", "reward_dics"
                    ],
                    num_train_steps=all_args.num_train_steps,
                    num_eval_episodes=all_args.num_eval_episodes,
                    eval_interval=all_args.eval_interval
                )
                trainer.train()

if __name__ == "__main__":
    main(sys.argv[1:])