import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_type", type=str,
                        choices=["planar"])
    parser.add_argument("--data_type", type=str,
                        choices=["mnist", "cifar10", "circle2d"])
    parser.add_argument("--config_json_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    return parser