import argparse
from argparse import ArgumentParser
import json
from train_hetero_rl import convert_str2path
from train_hetero_rl import create_env
from pandas import DataFrame
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
import sys
sys.path.append(str(root_path))
from drl_algos import Evaluater
from drl_algos import IPPO
from ots.evaluate_distances_real import create_ddevaluaters
import torch
from typing import Any
from typing import Optional

def get_config() -> ArgumentParser:
    parser: ArgumentParser = argparse.ArgumentParser()
    # for DDEvaluater
    parser.add_argument("--ohlcv_folder_path", type=str, default=None)
    parser.add_argument("--ticker_folder_names", type=str, nargs="*", default=None)
    parser.add_argument("--ticker_file_names", type=str, nargs="*", default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--resample_rule", type=str, default=None)
    parser.add_argument("--is_bybit", action="store_true")
    parser.add_argument("--lags", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--point_cloud_types", type=str, nargs="+",
        choices=["return", "tail_return", "rv_returns", "return_ts"]
    )
    # for algo, env
    parser.add_argument(
        "--algo_name", type=str, default="ippo", choices=["ippo"]
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--agent_name", type=str, default="heteroRLAgent")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--variable_ranges_path", type=str, required=False)
    parser.add_argument("--obs_names", type=str, nargs="+")
    parser.add_argument("--action_names", type=str, nargs="+")
    parser.add_argument("--depth_range", type=float, default=0.01)
    parser.add_argument("--limit_order_range", type=float, default=0.05)
    parser.add_argument("--max_order_volume", type=int, default=50)
    parser.add_argument("--short_selling_penalty", type=float, default=0.5)
    parser.add_argument("--cash_shortage_penalty", type=float, default=0.5)
    parser.add_argument("--initial_fundamental_penalty", type=float, default=10)
    parser.add_argument("--fundamental_penalty_decay", type=float, default=0.9)
    parser.add_argument("--execution_vonus", type=float, default=0.1)
    parser.add_argument("--agent_trait_memory", type=float, default=0.9)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    # for Evaluater
    parser.add_argument("--actor_folder_path", type=str, default=None)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.00, 0.005, 0.01, 0.015, 0.030]
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.00, 0.30, 0.60, 0.90, 2.00]
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95, 0.999]
    )
    parser.add_argument("--actor_seed", type=int, default=42)
    parser.add_argument("--txts_save_path", type=str, default=None)
    parser.add_argument("--tick_dfs_save_path", type=str, default=None)
    parser.add_argument("--ohlcv_dfs_save_path", type=str, default=None)
    parser.add_argument("--transactions_path", type=str, default=None)
    parser.add_argument("--session1_transactions_file_name", type=str, default=None)
    parser.add_argument("--session2_transactions_file_name", type=str, default=None)
    parser.add_argument("--market_name", type=str, default=None)
    parser.add_argument("--decision_histories_save_path", type=str, default=None)
    parser.add_argument("--figs_folder_path", type=str, default=None)
    parser.add_argument("--stylized_facts_folder_path", type=str, default=None)
    parser.add_argument("--ot_distances_save_path", type=str, default=None)
    return parser

def main(args) -> None:
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    dd_evaluaters = create_ddevaluaters(all_args, multiple_ts_only=True)
    config_path: Path = convert_str2path(all_args.config_path, mkdir=False)
    config_dic: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    env, num_agents = create_env(all_args, config_dic)
    env.fundamental_penalty = 0
    ippo: IPPO = IPPO(
        device=all_args.device,
        obs_shape=(len(all_args.obs_names),), action_shape=(len(all_args.action_names),),
        num_agents=num_agents,
        seed=all_args.seed,
    )
    stylized_facts_folder_path: Path = convert_str2path(all_args.stylized_facts_folder_path, True)
    figs_folder_path: Path = convert_str2path(all_args.figs_folder_path, True)
    actor_folder_path: Path = convert_str2path(all_args.actor_folder_path, False)
    sigmas: list[float] = all_args.sigmas
    alphas: list[float] = all_args.alphas
    gammas: list[float] = all_args.gammas
    for sigma in sigmas:
        sigma_str: str = f"{sigma:.3f}".replace(".", "")
        config_dic["Agent"]["skillBoundedness"] = {"expon": [sigma]}
        for alpha in alphas:
            alpha_str: str = f"{alpha:.2f}".replace(".", "")
            config_dic["Agent"]["riskAversionTerm"] = {"expon": [alpha]}
            for gamma in gammas:
                gamma_str: str = f"{gamma:.2f}".replace(".", "")
                config_dic["Agent"]["discountFactor"] = {"uniform": [gamma, 0.999]}
                print(f"agent config: {config_dic['Agent']}")
                actor_name = f"best-{sigma_str}-{alpha_str}-{gamma_str}-{all_args.actor_seed}"
                actor_load_path: Path = actor_folder_path / f"{actor_name}.pth"
                stylized_facts_save_path: Path = stylized_facts_folder_path / f"{actor_name}.csv"
                figs_save_path: Path = figs_folder_path / actor_name
                evaluater: Evaluater = Evaluater(
                    dd_evaluaters=dd_evaluaters, num_points=1500,
                    env=env, algo=ippo, actor_load_path=actor_load_path,
                    txts_save_path=convert_str2path(all_args.txts_save_path, True),
                    tick_dfs_save_path=convert_str2path(all_args.tick_dfs_save_path, True),
                    ohlcv_dfs_save_path=convert_str2path(all_args.ohlcv_dfs_save_path, True),
                    transactions_path=convert_str2path(all_args.transactions_path, False),
                    session1_transactions_file_name=all_args.session1_transactions_file_name,
                    session2_transactions_file_name=all_args.session2_transactions_file_name,
                    market_name=all_args.market_name,
                    decision_histories_save_path=convert_str2path(all_args.decision_histories_save_path, True),
                    figs_save_path=figs_save_path,
                    stylized_facts_save_path=stylized_facts_save_path,
                    ot_distances_save_path=convert_str2path(all_args.ot_distances_save_path, False),
                )
                decision_histories_dfs: list[DataFrame] = evaluater.save_multiple_episodes(
                    start_num=0, episode_num=300, unlink_all=True
                )
                evaluater.hist_obs_actions(
                    decision_histories_dfs=decision_histories_dfs,
                    obs_save_name="hist_obs.pdf",
                    action_save_name="hist_actions.pdf"
                )
                # evaluater.scatter_pl_given_agent_trait(
                #     decision_histories_dfs=decision_histories_dfs,
                #     trait_column_names=["skill_boundedness", "risk_aversion_term", "discount_factor"],
                #     save_names=[
                #         "scatter_sigma_pl.pdf",
                #         "scatter_alpha_pl.pdf",
                #         "scatter_gamma_pl.pdf"
                #     ]
                # )
                # evaluater.scatter_price_range_given_agent_trait(
                #     decision_histories_dfs=decision_histories_dfs,
                #     trait_column_names=["skill_boundedness", "risk_aversion_term", "discount_factor"],
                #     save_names=[
                #         "scatter_sigma_order_price_scale.pdf",
                #         "scatter_alpha_order_price_scale.pdf",
                #         "scatter_gamma_order_price_scale.pdf"
                #     ]
                # )
                evaluater.draw_actions_given_obs2d(
                    target_obs_names=[r"risk aversion term $\alpha^j$", r"volatility $V_{[t_{i-1}^j,t_i^j]}$"],
                    target_obs_indices=[9, 4],
                    target_action_idx=1,
                    x_obs_values=[-0.95+0.1*x for x in range(19)],
                    y_obs_values=[-0.95+0.1*x for x in range(19)],
                    initial_obs_values=[
                        -0.30, -0.80, -0.80, 0.00, -0.80,
                        -0.80, -0.80, 0.00, -0.80, -0.80, 0.00
                    ],
                    save_name="heatmap_order_volume_given_alpha_volatility.pdf",
                )
                evaluater.draw_actions_given_obs2d(
                    target_obs_names=[r"skill boundedness $\sigma^j$", r"blurred fundamental return $\tilde{r}_t^f$"],
                    target_obs_indices=[10, 7],
                    target_action_idx=1,
                    x_obs_values=[-0.95+0.1*x for x in range(19)],
                    y_obs_values=[-0.05+0.005*x for x in range(19)],
                    initial_obs_values=[
                        -0.30, -0.80, -0.80, 0.00, -0.80,
                        -0.80, -0.80, 0.00, -0.80, -0.80, 0.00
                    ],
                    save_name="heatmap_order_volume_given_sigma_rf.pdf",
                )
                evaluater.draw_actions_given_obs2d(
                    target_obs_names=[r"skill boundedness $\sigma^j$", r"log return $r_{[t_{i-1}^j,t_i^j]}$"],
                    target_obs_indices=[10, 3],
                    target_action_idx=1,
                    x_obs_values=[-0.95+0.1*x for x in range(19)],
                    y_obs_values=[-0.05+0.005*x for x in range(19)],
                    initial_obs_values=[
                        -0.30, -0.80, -0.80, 0.00, -0.80,
                        -0.80, -0.80, 0.00, -0.80, -0.80, 0.00
                    ],
                    save_name="heatmap_order_volume_given_sigma_r.pdf",
                )
                evaluater.draw_actions_given_obs1d(
                    target_obs_name=r"discount factor $\gamma^j$",
                    target_obs_idx=10,
                    target_action_name=r"order price scale $\tilde{r}_t^j$",
                    target_action_idx=0,
                    obs_values=[-0.99+0.01*x for x in range(199)],
                    initial_obs_values=[
                        -0.30, -0.80, -0.80, 0.00, -0.80,
                        -0.80, -0.80, 0.00, -0.80, -0.80, 0.00
                    ],
                    save_name="plot_order_price_given_gamma.pdf",
                )
                


if __name__ == "__main__":
    main(sys.argv[1:])