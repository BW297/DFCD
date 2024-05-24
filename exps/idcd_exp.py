import os
import sys
import argparse
import torch
from pprint import pprint


def import_paths():
    current_path = os.path.abspath('.')
    tmp = os.path.dirname(current_path)
    sys.path.insert(0, tmp)
    sys.path.insert(0, tmp + '/models')


import_paths()

from models.idcd import IDCD
from utils import load_data, get_interaction_matrix, set_common_args


def main(config):
    # 加载数据
    load_data(config)

    if config['split'] == 'Stu' or config['split'] == 'Exer':
        full_interaction_matrix, train_interaction_matrix = torch.tensor(
        get_interaction_matrix(config, data=config['np_train'])).to(config['device']), torch.tensor(
        get_interaction_matrix(config, data=config['np_train_old'])).to(config['device'])
    else:
        full_interaction_matrix, train_interaction_matrix = torch.tensor(
        get_interaction_matrix(config, data=config['np_train'])).to(config['device']), torch.tensor(
        get_interaction_matrix(config, data=config['np_train'])).to(config['device'])
    config['train_interaction_matrix'], config[
        'full_interaction_matrix'] = train_interaction_matrix, full_interaction_matrix

    # 创建IDCD模型实例
    idcd = IDCD(config)

    # 训练模型
    idcd.train_step()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    set_common_args(parser)
    config_dict = vars(parser.parse_args())
    name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"
    config_dict['name'] = name

    # 打印配置信息
    pprint(config_dict)
    # 执行主函数
    main(config_dict)
