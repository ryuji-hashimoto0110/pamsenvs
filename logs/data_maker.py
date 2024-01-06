import pathlib
from pathlib import Path
import sys
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
sys.path.append(curr_path)
from logs.volumeprice_logger import VolumePriceSaver
from pams.runners.base import Runner
from pams.runners.sequential import SequentialRunner
import random
from typing import Any
from typing import Optional
from typing import TypeVar

MarketID = TypeVar("MarketID")

class DataMaker:
    def create_artificial_olhcvs(
        self,
        config: Path | dict[str, Any],
        datas_path: Path,
        market_id: MarketID,
        start_index: int,
        index_interval: int,
        data_num: int,
        seeds: Optional[list[int]] = None
    ) -> None:
        if seeds is None:
            seeds: list[int] = [i+1 for i in range(data_num)]
        assert len(seeds) == data_num
        if not datas_path.exists():
            datas_path.mkdir(parents=True)
        for seed in seeds:
            saver = VolumePriceSaver()
            try:
                runner = SequentialRunner(
                    settings=config,
                    prng=random.Random(seed),
                    logger=saver,
                )
                self.class_register(runner)
                runner.main()
                save_path: Path = datas_path / f"{seed}.csv"
                saver.save_olhcv(
                    market_id, start_index, index_interval, save_path
                )
            except Exception as e:
                print(f"seed{seed}: {e}")

    def class_register(self, runner: Runner):
        pass

