import torch
import torch.nn as nn
from utils import NoneNegClipper


def get_decoder(config):
    if config['decoder_type'] == 'simplecd':
        return SimpleCDDecoder(config)
    elif config['decoder_type'] == 'kancd':
        return KaNCDDecoder(config)
    elif config['decoder_type'] == 'ncd':
        return NCDDecoder(config)
    else:
        raise ValueError('Unexplored')


def Positive_MLP(config, num_layers=3, hidden_dim=512, dropout=0.5):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(config['know_num'] if i == 0 else hidden_dim // pow(2, i - 1),
                                hidden_dim // pow(2, i)))
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Tanh())

    layers.append(nn.Linear(hidden_dim // pow(2, num_layers - 1), 1))
    layers.append(nn.Sigmoid())
    layers = nn.Sequential(*layers)
    return layers


class NCDDecoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.layers = Positive_MLP(config).to(config['device'])
        self.transfer_student_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.transfer_exercise_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        state = knowledge_point * (torch.sigmoid(self.transfer_student_layer(z[student_id])) - torch.sigmoid(
            self.transfer_exercise_layer(z[exercise_id])))
        return self.layers.forward(state).view(-1)

    def get_mastery_level(self, z):
        return torch.sigmoid(self.transfer_student_layer(z[:self.config['stu_num']])).detach().cpu().numpy()

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

class KaNCDDecoder(nn.Module):

    def __init__(
            self, config,
    ):
        super().__init__()
        self.k_diff_full = nn.Linear(config['out_channels'], 1).to(config['device'])
        self.stat_full = nn.Linear(config['out_channels'], 1).to(config['device'])
        self.layers = Positive_MLP(config).to(config['device'])
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        knowledge_ts = z[self.config['stu_num'] + self.config['prob_num']:]
        stu_emb = z[student_id]
        exer_emb = z[exercise_id]
        dim = z.shape[1]
        batch = student_id.shape[0]

        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.config['know_num'], 1)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.config['know_num'], -1)
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.config['know_num'], 1)
        stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        state = knowledge_point * (stat_emb - k_difficulty)
        return self.layers.forward(state).view(-1)

    def get_mastery_level(self, z):
        knowledge_ts = z[self.config['stu_num'] + self.config['prob_num']:]
        return torch.sigmoid(z[:self.config['stu_num']] @ knowledge_ts.T).detach().cpu().numpy()

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


class SimpleCDDecoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.layers = Positive_MLP(config).to(config['device'])
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        knowledge_ts = z[self.config['stu_num'] + self.config['prob_num']:]
        state = knowledge_point * (torch.sigmoid(z[student_id] @ knowledge_ts.T) - torch.sigmoid(
            z[exercise_id + self.config['stu_num']] @ knowledge_ts.T))
        return self.layers.forward(state).view(-1)

    def get_mastery_level(self, z):
        knowledge_ts = z[self.config['stu_num'] + self.config['prob_num']:]
        return torch.sigmoid(z[:self.config['stu_num']] @ knowledge_ts.T).detach().cpu().numpy()

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


def create_gnn_encoder(encoder_type, in_channels, out_channels, num_heads=4):
    from torch_geometric.nn import (
        GCNConv,
        GATConv,
        GATv2Conv,
        TransformerConv,
    )
    if encoder_type == 'gat':
        return GATConv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    elif encoder_type == 'gatv2':
        return GATv2Conv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    elif encoder_type == 'gcn':
        return GCNConv(in_channels=in_channels, out_channels=out_channels)
    elif encoder_type == 'transformer':
        return TransformerConv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    else:
        raise ValueError('Unexplored')

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

def get_mlp_encoder(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, 512),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, out_channels),
    )

def to_sparse_tensor(edge_index, num_nodes):
    from torch_sparse import SparseTensor
    return SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to(edge_index.device)

class GNNEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=2,
            dropout=0.5,
            bn=False,
            layer='gcn',
            activation="elu",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer or 'transformer' not in layer else 4

            self.convs.append(create_gnn_encoder(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels * heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = to_sparse_tensor(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat"):

        self.eval()
        assert mode in {"cat", "last"}, mode

        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding
