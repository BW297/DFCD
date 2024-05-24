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

from models.ncdm import NCDM
from utils import load_data, set_common_args, get_interaction_matrix, construct_data_geometric



def main(config):
    # 加载数据
    load_data(config)
    train_data = construct_data_geometric(config, data=config['np_train'])
    full_data = construct_data_geometric(config, data=config['np_train'])
    config['train_data'] = train_data.to(config['device'])
    config['full_data'] = full_data.to(config['device'])

    if config["text_embedding_model"] == "openai":
        config['in_channels_llm'] = 1536
    elif config["text_embedding_model"] == "BAAI":
        config['in_channels_llm'] = 1024
    elif config["text_embedding_model"] == "m3e":
        config['in_channels_llm'] = 768
    elif config["text_embedding_model"] == "instructor":
        config['in_channels_llm'] = 768
    # 创建IDCD模型实例
    ncdm = NCDM(config)

    # 训练模型
    ncdm.train_step()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', default='normal')
    set_common_args(parser)
    config_dict = vars(parser.parse_args())
    name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"
    config_dict['name'] = name

    # 打印配置信息
    pprint(config_dict)
    # 执行主函数
    main(config_dict)
