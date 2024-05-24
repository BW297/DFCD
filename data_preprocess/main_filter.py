import argparse
import importlib


def params_parser(args):
    config = dict()
    config['dataset'] = args.dataset
    config['stu_num'] = args.stu_num
    config['exer_num'] = args.exer_num
    config['know_num'] = args.know_num
    config['least_respone_num'] = args.least_respone_num
    config['seed'] = args.seed

    method = importlib.import_module(f'{args.dataset}.preprocess')
    method.run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['XES3G5M', 'NeurIPS2020', 'MOOCRadar'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stu_num', type=int, default=4000)
    parser.add_argument('--exer_num', type=int, default=2000)
    parser.add_argument('--know_num', type=int, default=300)
    parser.add_argument('--least_respone_num', type=int, default=50)
    args = parser.parse_args()
    params_parser(args)
