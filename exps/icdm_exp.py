import os
import sys
import argparse
import torch
from pprint import pprint


def import_paths():
    import warnings
    warnings.filterwarnings("ignore")
    current_path = os.path.abspath('.')
    tmp = os.path.dirname(current_path)
    sys.path.insert(0, tmp)
    sys.path.insert(0, tmp + '/models')


import_paths()

from models.icdm import ICDM
from utils import load_data, set_common_args, build_graph4CE, build_graph4SC, build_graph4SE
def main(config):
    # 加载数据
    load_data(config)
    if config['split'] == 'Stu':
        right_old, wrong_old = build_graph4SE(config, mode='ind_train')
        right_eval, wrong_eval = build_graph4SE(config, mode='ind_eval')
        config['right_old'] = right_old
        config['wrong_old'] = wrong_old
        config['right_eval'] = right_eval
        config['wrong_eval'] = wrong_eval
        config['Q'] = build_graph4CE(config)
        config['I_old'] = build_graph4SC(config, mode='ind_train')
        config['I_eval'] = build_graph4SC(config, mode='ind_eval')
        config['involve'] = build_graph4SC(config, mode='involve')
    else:
        right_old, wrong_old = build_graph4SE(config, mode='tl')
        right_eval, wrong_eval = build_graph4SE(config, mode='tl')
        config['exist_idx'] = torch.arange(config['stu_num'])
        config['right_old'] = right_old
        config['wrong_old'] = wrong_old
        config['right_eval'] = right_eval
        config['wrong_eval'] = wrong_eval
        config['Q'] = build_graph4CE(config)
        config['I_old'] = build_graph4SC(config, mode='tl')
        config['I_eval'] = build_graph4SC(config, mode='tl')
        config['involve'] = build_graph4SC(config, mode='tl')

    # 创建ICDM模型实例
    icdm = ICDM(config)

    # 训练模型
    icdm.train_step()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    set_common_args(parser)
    parser.add_argument('--dim', default=32)
    parser.add_argument('--cdm_type', default='glif')
    parser.add_argument('--agg_type', default='mean')
    parser.add_argument('--khop', default=3)
    parser.add_argument('--gcn_layers', default=3)
    parser.add_argument('--d_1', default=0.1)
    parser.add_argument('--d_2', default=0.2)
    config_dict = vars(parser.parse_args())
    name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"
    config_dict['name'] = name

    # 打印配置信息
    pprint(config_dict)
    # 执行主函数
    main(config_dict)
