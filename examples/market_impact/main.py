import argparse
import pathlib
import random
import sys
curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
parent_path: pathlib.Path = curr_path.parents[0]
grandparent_path: pathlib.Path = curr_path.parents[2]
sys.path.append(str(curr_path))
sys.path.append(str(grandparent_path))
from logs import OrderBookSaver
from whale_agent import WhaleAgent
from pams.runners.sequential import SequentialRunner
from typing import Optional
import warnings
warnings.simplefilter('ignore')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="config.json file"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="simulation random seed"
    )
    args = parser.parse_args()
    config: str = args.config
    seed: Optional[int] = args.seed
    videos_path = grandparent_path / "videos"
    orderbook_saver = OrderBookSaver(videos_path, specific_agent_color_dic={100: "tab:green"},
                                    video_width=2700, video_height=2700)
    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        logger=orderbook_saver,
    )
    runner.class_register(WhaleAgent)
    runner.main()

if __name__ == "__main__":
    main()