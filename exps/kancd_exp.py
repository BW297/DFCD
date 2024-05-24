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

from models.kancd import KaNCD
from utils import load_data, set_common_args, get_interaction_matrix, find_nearest_neighbors, construct_data_geometric


def get_nearest_dict(config):
    interaction_matrix = get_interaction_matrix(config, config['np_data'])
    if config['split'] == 'Stu':
        return find_nearest_neighbors(interaction_matrix)
    else:
        return find_nearest_neighbors(interaction_matrix.T)



def main(config):
    # 加载数据
    load_data(config)
    train_data = construct_data_geometric(config, data=config['np_train'])
    full_data = construct_data_geometric(config, data=config['np_train'])
    config['train_data'] = train_data.to(config['device'])
    config['full_data'] = full_data.to(config['device'])

    config['nearest'] = get_nearest_dict(config)
    if config["text_embedding_model"] == "openai":
        config['in_channels_llm'] = 1536
    elif config["text_embedding_model"] == "BAAI":
        config['in_channels_llm'] = 1024
    elif config["text_embedding_model"] == "m3e":
        config['in_channels_llm'] = 768
    elif config["text_embedding_model"] == "instructor":
        config['in_channels_llm'] = 768
    # 创建IDCD模型实例
    kancd = KaNCD(config)

    # 训练模型
    kancd.train_step()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=20)
    parser.add_argument('--embedding_method', default='Nearest')
    parser.add_argument('--exp_type', default='normal')
    set_common_args(parser)
    config_dict = vars(parser.parse_args())
    name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"
    config_dict['name'] = name

    # 打印配置信息
    pprint(config_dict)
    # 执行主函数
    main(config_dict)
