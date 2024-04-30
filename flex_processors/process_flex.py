import argparse
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from flex_processors import FlexProcessor
import sys

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_folder_path", type=str, default=None,
                        help="folder path that target txt datas are stored.")
    parser.add_argument("--csv_folder_path", type=str, default=None,
                        help="folder path that target txt datas are stored.")
    parser.add_argument("--quote_num", type=int)
    return parser

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    txt_folder: str = all_args.txt_folder_path
    txt_datas_path: Path = pathlib.Path(txt_folder)
    csv_folder: str = all_args.csv_folder_path
    csv_datas_path: Path = pathlib.Path(csv_folder)
    if not csv_datas_path.exists():
        csv_datas_path.mkdir(parents=True)
    quote_num: int = all_args.quote_num
    processor = FlexProcessor(
        txt_datas_path=txt_datas_path,
        csv_datas_path=csv_datas_path,
        quote_num=quote_num,
        is_execution_only=True
    )
    processor.convert_all_txt2csv()

if __name__ == "__main__":
    main(sys.argv[1:])