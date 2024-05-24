import argparse
import importlib


def params_parser(args):
    config = dict()
    config['dataset'] = args.dataset
    config['llm'] = args.llm

    method = importlib.import_module(f'{args.dataset}.embedding')
    method.run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['XES3G5M', 'NeurIPS2020', 'MOOCRadar'], required=True)
    parser.add_argument('--llm', type=str, default='OpenAI')
    args = parser.parse_args()
    params_parser(args)
