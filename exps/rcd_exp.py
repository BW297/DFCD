import torch
import numpy as np
import argparse
import sys
import os
from pprint import pprint
def import_paths():
    import warnings
    warnings.filterwarnings("ignore")
    current_path = os.path.abspath('.')
    tmp = os.path.dirname(current_path)
    sys.path.insert(0, tmp)
    sys.path.insert(0, tmp + '/instant_cd')


import_paths()


from inscd.datahub import DataHub
from inscd.models.static.graph import RCD


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--method', default='rcd', type=str,
                    help='A Lightweight Graph-based Cognitive Diagnosis Framework', required=True)
parser.add_argument('--data_type', default='junyi', type=str, help='benchmark', required=True)
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark', required=True)
parser.add_argument('--epoch', type=int, help='epoch of method', default=20)
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=True)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cuda', type=str, help='device for exp')
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=1024)
parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0)
config_dict = vars(parser.parse_args())

name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"

pprint(config_dict)

def main(config):
    set_seeds(config['seed'])
    datahub = DataHub(f"../data/{config['data_type']}")
    datahub.random_split(source="total", to=["train", "test"], seed=config['seed'], slice_out=1 - config['test_size'])
    validate_metrics = ['auc', 'acc', 'ap', 'rmse', 'f1', 'doa']
    print("Number of response logs {}".format(len(datahub)))
    rcd = RCD(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
    rcd.build(device=config['device'], if_type='rcd', dtype=torch.float64)
    rcd.train(datahub, "train", "test", valid_metrics=validate_metrics, lr=3e-3,
                  batch_size=config['batch_size'], weight_decay=0)


if __name__ == '__main__':
    sys.exit(main(config_dict))
