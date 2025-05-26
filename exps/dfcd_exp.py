import os
import sys
import argparse
from pprint import pprint


def import_paths():
    import warnings
    warnings.filterwarnings("ignore")
    current_path = os.path.abspath('.')
    tmp = os.path.dirname(current_path)
    sys.path.insert(0, tmp)
    sys.path.insert(0, tmp + '/models')


import_paths()

from models.dfcd import DFCD
from utils import load_data, set_common_args, construct_data_geometric
def main(config):
    # 加载数据
    load_data(config)
    if config["text_embedding_model"] == "openai":
        config['in_channels_llm'] = 1536
    elif config["text_embedding_model"] == "BAAI":
        config['in_channels_llm'] = 1024
    elif config["text_embedding_model"] == "m3e":
        config['in_channels_llm'] = 768
    elif config["text_embedding_model"] == "instructor":
        config['in_channels_llm'] = 768
    config['in_channels_init'] = config['stu_num'] + config['prob_num'] + config['know_num']
    if config['split'] == 'Stu' or config['split'] == 'Exer':
        train_data = construct_data_geometric(config, data=config['np_train_old'])
        full_data = construct_data_geometric(config, data=config['np_train'])
    else:
        train_data = construct_data_geometric(config, data=config['np_train'])
        full_data = construct_data_geometric(config, data=config['np_train'])

    config['train_data'] = train_data.to(config['device'])
    config['full_data'] = full_data.to(config['device'])
    dfcd = DFCD(config)
    # 训练模型
    dfcd.train_step()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default='transformer', type=str)
    parser.add_argument('--decoder_type', default='simplecd', type=str)
    parser.add_argument('--out_channels', default=128, type=int)
    parser.add_argument('--mode', default=2, type=int)
    set_common_args(parser)
    config_dict = vars(parser.parse_args())
    name = f"{config_dict['method']}-{config_dict['data_type']}-seed{config_dict['seed']}"
    config_dict['name'] = name
    if config_dict['mode'] == 1:
        config_dict['method'] = config_dict['method'] + '-text'
    elif config_dict['mode'] == 2:
        config_dict['method'] = config_dict['method'] + '-hybrid'
    elif config_dict['mode'] == 0:
        config_dict['method'] = config_dict['method'] + '-response'

    # 打印配置信息
    pprint(config_dict)
    # 执行主函数
    main(config_dict)