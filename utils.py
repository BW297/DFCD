import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def transform(q: torch.tensor, user, item, score, batch_size, dtype=torch.float64):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_data(config):
    set_seed(config['seed'])
    torch.set_default_dtype(config['dtype'])
    device_id = config['device'].split(':')[1]
    torch.cuda.set_device(int(device_id))
    from data.data_params_dict import data_params
    config.update({
        'stu_num': data_params[config["data_type"]]['stu_num'],
        'prob_num': data_params[config["data_type"]]['prob_num'],
        'know_num': data_params[config["data_type"]]['know_num'],
    })

    import pickle
    with open('../data/{}/embedding_{}.pkl'.format(config["data_type"], config["text_embedding_model"]), 'rb') as file:
        embeddings = pickle.load(file)
        config['s_embed'] = embeddings['student_embeddings']
        config['e_embed'] = embeddings['exercise_embeddings']
        config['k_embed'] = embeddings['knowledge_embeddings']
    # Load q.csv
    q_np = pd.read_csv(f'../data/{config["data_type"]}/q.csv', header=None).to_numpy()
    q_tensor = torch.tensor(q_np).to(config['device'])

    # Load TotalData.csv
    np_data = pd.read_csv(f'../data/{config["data_type"]}/TotalData.csv', header=None).to_numpy()

    TotalData = pd.DataFrame(np_data, columns=['stu', 'exer', 'answervalue'])
    se_map = {}
    for stu_id in range(config['stu_num']):
        student_logs = TotalData.loc[TotalData['stu'] == stu_id]
        se_map[stu_id] = {}
        cnt = 0
        for log in student_logs.values:
            se_map[stu_id][log[1]] = cnt
            cnt += 1
    config['se_map'] = se_map

    np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    if config['split'] == 'Stu':
        train_stu = np.random.choice(np.arange(config['stu_num']),
                                     size=int((1 - config['test_size']) * config['stu_num']),
                                     replace=False)
        config['exist_idx'] = train_stu.astype(np.int64)
        config['np_train_old'] = np_train[np.isin(np_train[:, 0], config['exist_idx'])]
        config['np_train_new'] = np_train[~np.isin(np_train[:, 0], config['exist_idx'])]
        config['new_idx'] = np.setdiff1d(np.arange(config['stu_num']), config['exist_idx']).tolist()

    elif config['split'] == 'Exer':
        train_exer = np.random.choice(np.arange(config['prob_num']),
                                      size=int((1 - config['test_size']) * config['prob_num']),
                                      replace=False)
        config['exist_idx'] = train_exer.astype(np.int64)
        config['np_train_old'] = np_train[np.isin(np_train[:, 1], config['exist_idx'])]
        config['np_train_new'] = np_train[~np.isin(np_train[:, 1], config['exist_idx'])]
        config['new_idx'] = np.setdiff1d(np.arange(config['prob_num']), train_exer).tolist()
    elif config['split'] == 'Know':
        train_know = np.random.choice(np.arange(config['know_num']), 
                                       size=int((1 - config['test_size']) * config['know_num']),
                                       replace=False)
        train_exer = np.where((q_np[:, train_know] == 1).any(axis=1))[0]
        config['exist_idx'] = train_exer.astype(np.int64)
        config['np_train_old'] = np_train[np.isin(np_train[:, 1], config['exist_idx'])]
        config['np_train_new'] = np_train[~np.isin(np_train[:, 1], config['exist_idx'])]
        config['new_idx'] = np.setdiff1d(np.arange(config['prob_num']), train_exer).tolist()
    else:
        np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])

    # Update config with loaded data
    config.update({
        'np_data': np_data,
        'np_train': np_train,
        'np_test': np_test,
        'q': q_tensor,
        'r': get_r_matrix(np_test, config['stu_num'], config['prob_num'])
    })

    config['train_dataloader'], config['test_dataloader'] = get_dataloader(config)


def get_dataloader(config):
    if config['split'] == 'Stu' or config['split'] == 'Exer' or config['split'] == 'Know':
        train_dataloader, test_dataloader = [
        transform(config['q'], _[:, 0], _[:, 1], _[:, 2], config['batch_size'])
        for _ in [config['np_train_old'], config['np_test']]]
    else:
        train_dataloader, test_dataloader = [
        transform(config['q'], _[:, 0], _[:, 1], _[:, 2], config['batch_size'])
        for _ in [config['np_train'], config['np_test']]]
    return train_dataloader, test_dataloader

def get_top_k_concepts(datatype: str, topk: int = 10):
    q = pd.read_csv('../data/{}/q.csv'.format(datatype), header=None).to_numpy()
    a = pd.read_csv('../data/{}/TotalData.csv'.format(datatype), header=None).to_numpy()
    skill_dict = {}
    for k in range(q.shape[1]):
        skill_dict[k] = 0
    for k in range(a.shape[0]):
        stu_id = a[k, 0]
        prob_id = a[k, 1]
        skills = np.where(q[int(prob_id), :] != 0)[0].tolist()
        for skill in skills:
            skill_dict[skill] += 1

    sorted_dict = dict(sorted(skill_dict.items(), key=lambda x: x[1], reverse=True))
    all_list = list(sorted_dict.keys())  # 189
    return all_list[:topk]


def get_doa(config, mastery_level):
    q_matrix = config['q'].detach().cpu().numpy()
    r_matrix = config['r']
    from metrics.DOA import calculate_doa_k_block
    check_concepts = get_top_k_concepts(config['data_type'])
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in check_concepts)
    return np.mean(doa_k_list)


def set_common_args(parser):
    parser.add_argument('--method', default='idcd', type=str, help='')
    parser.add_argument('--data_type', default='NeurIPS2020', type=str, help='benchmark')
    parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark')
    parser.add_argument('--epoch', default=20, type=int, help='epoch of method')
    parser.add_argument('--seed', default=0, type=int, help='seed for exp')
    parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
    parser.add_argument('--device', default='cuda:0', type=str, help='device for exp')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training. (default: 1e-2)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of batch size. (default: 2**16)')
    parser.add_argument('--split', default='Stu', type=str, help='Split Method')
    parser.add_argument('--text_embedding_model', default='openai', type=str, help='text embedding model')
    parser.add_argument('--weight_decay', default=0, type=float)
    return parser


def get_r_matrix(np_test, stu_num, prob_num, new_idx=None):
    if new_idx is None:
        r = -1 * np.ones(shape=(stu_num, prob_num))
        for i in range(np_test.shape[0]):
            s = int(np_test[i, 0])
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    else:
        r = -1 * np.ones(shape=(stu_num, prob_num))

        for i in range(np_test.shape[0]):
            s = new_idx.index(int(np_test[i, 0]))
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    return r


def construct_data_geometric(config, data):
    from torch_geometric.data import Data

    sek = config['stu_num'] + config['prob_num'] + config['know_num']

    se_source_index, ek_source_index = [], []
    se_target_index, ek_target_index = [], []
    se_label = []
    s_train_embed = []
    
    TotalData = pd.DataFrame(config['np_train'], columns=['stu', 'exer', 'answervalue'])
    for stu_id in range(config['stu_num']):
        student_logs = TotalData.loc[TotalData['stu'] == stu_id]
        tmp_embed = []
        for log in student_logs.values:
            tmp_embed.append(config['s_embed'][stu_id][config['se_map'][stu_id][log[1]]])
        s_train_embed.append(tmp_embed)


    node_features_llm = torch.tensor(
        [np.array(s_train_embed[stu_id]).mean(axis=0) for stu_id in range(config['stu_num'])] +
        [config['e_embed'][exer_id] for exer_id in range(config['prob_num'])] +
        [config['k_embed'][know_id] for know_id in range(config['know_num'])],
        dtype=torch.float64)

    node_features_init = torch.zeros(size=(sek, sek), dtype=torch.float64)
    for _, (stu_id, exer_id, label) in enumerate(data):
        stu_id, exer_id = int(stu_id), int(exer_id)
        if label == 1:
            node_features_init[stu_id, exer_id + config['stu_num']] = 1
            node_features_init[exer_id + config['stu_num'], stu_id] = 1
        else:
            node_features_init[stu_id, exer_id + config['stu_num']] = -1
            node_features_init[exer_id + config['stu_num'], stu_id] = -1

    for _, (stu_id, exer_id, label) in enumerate(data):
        stu_id, exer_id = int(stu_id), int(exer_id)
        se_source_index.append(stu_id)
        se_target_index.append(exer_id + config['stu_num'])
        se_label.append(label)

    for exer_id, know_id in zip(*np.where(config['q'].detach().cpu().numpy() != 0)):
        ek_source_index.append(exer_id + config['stu_num'])
        ek_target_index.append(know_id + config['stu_num'] + config['prob_num'])

    edge_index = torch.tensor([se_source_index + ek_source_index, se_target_index + ek_target_index], dtype=torch.long)
    graph_data = Data(x_llm=node_features_llm, x_init=node_features_init, edge_index=edge_index)
    return graph_data

class Weighted_Summation(nn.Module):
    def __init__(self, hidden_dim, attn_drop, dtype=torch.float64):
        super(Weighted_Summation, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=dtype), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc