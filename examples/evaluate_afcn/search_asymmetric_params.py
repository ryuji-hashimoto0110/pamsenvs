import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
root_path: Path = curr_path.parents[1]
config_path: Path = root_path / "examples" / "evaluate_afcn" / "afcn_config.json"
import sys
sys.path.append(str(root_path))
from envs.agents import aFCNAgent
import json
from logs import DataMaker
from pams.runners.base import Runner
from typing import Any
from typing import TypeVar

MarketID = TypeVar("MarketID")
parent_daily_datas_path: Path = root_path / "datas" / "artificial_datas" / "daily" / "afcn"
parent_intraday_datas_path: Path = root_path / "datas" / "artificial_datas" / "intraday" / "afcn"

class aFCNDataMaker(DataMaker):
    def class_register(self, runner: Runner):
        runner.class_register(aFCNAgent)

    def create_artificial_olhcvs_w_various_asymmetry_params(
        self,
        a_feedbacks: list[int | float],
        a_noises: list[int | float],
        config: Path | dict[str, Any]
    ) -> None:
        for a_feedback in a_feedbacks:
            for a_noise in a_noises:
                config["aFCNAgents"]["feedbackAsymmetry"]["expon"] = [a_feedback]
                config["aFCNAgents"]["noiseAsymmetry"]["expon"] = [a_noise]
                daily_datas_path: Path = parent_daily_datas_path / \
                    f"af{str(a_feedback).replace('.','')}_an{str(a_noise).replace('.','')}"
                if not daily_datas_path.exists():
                    daily_datas_path.mkdir(parents=True)
                intraday_datas_path: Path = parent_intraday_datas_path / \
                    f"af{str(a_feedback).replace('.','')}_an{str(a_noise).replace('.','')}"
                if not intraday_datas_path.exists():
                    intraday_datas_path.mkdir(parents=True)
                self.create_artificial_olhcvs(
                    config, daily_datas_path, intraday_datas_path, 0, 1000, 72, 1, 100
                )

if __name__ == "__main__":
    config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    a_feedbacks: list[float] = [3]
    a_noises: list[float] = [0]
    data_maker = aFCNDataMaker()
    data_maker.create_artificial_olhcvs_w_various_asymmetry_params(
        a_feedbacks, a_noises, config
    )
