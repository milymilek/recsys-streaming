import argparse
import importlib


SCRIPT_MAP = {
    'download_data': ('recsys_streaming_ml.data', 'download_data'),
    'train': ('recsys_streaming_ml.model', 'train'),
    'evaluate': ('recsys_streaming_ml.model', 'evaluate')
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run a specific script.')
    parser.add_argument("-s", '--script', choices=['download_data', 'train', 'evaluate'], help='Specify the script to run')
    return parser.parse_args()


def main():
    args = parse_args()

    module_name, function_name = SCRIPT_MAP[args.script]

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    function()


if __name__ == "__main__":
    main()