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
from logs import VolumePriceSaver
from pams.runners.base import Runner
from pams.runners.sequential import SequentialRunner
import random
from typing import Any
from typing import Optional
from typing import TypeVar

MarketID = TypeVar("MarketID")
parent_datas_path: Path = root_path / "datas" / "artificial_datas" / "afcn"

class aFCNDataMaker(DataMaker):
    def class_register(self, runner: Runner):
        runner.class_register(aFCNAgent)

    def create_artificial_olhcvs_w_various_asymmetry_params(
        self,
        a_feedbacks: list[int],
        a_noises: list[int],
        config: Path | dict[str, Any]
    ) -> None:
        for a_feedback in a_feedbacks:
            for a_noise in a_noises:
                config["aFCNAgents"]["feedbackAsymmetry"]["expon"] = [a_feedback]
                config["aFCNAgents"]["noiseAsymmetry"]["expon"] = [a_noise]
                datas_path: Path = parent_datas_path / f"af{a_feedback}_an{a_noise}"
                if not datas_path.exists():
                    datas_path.mkdir(parents=True)
                self.create_artificial_olhcvs(
                    config, datas_path, 0, 100, 10, 10
                )

if __name__ == "__main__":
    config: dict[str, Any] = json.load(fp=open(str(config_path), mode="r"))
    a_feedbacks: list[float] = [0, 10, 20, 30, 40, 50]
    a_noises: list[float] = [0, 10, 20, 30, 40, 50]
    data_maker = aFCNDataMaker()
    data_maker.create_artificial_olhcvs_w_various_asymmetry_params(
        a_feedbacks, a_noises, config
    )
