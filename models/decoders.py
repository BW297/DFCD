import torch
import torch.nn as nn
from utils import NoneNegClipper



"""
Net for LLM-CD
"""
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
        Linear,
        GCNConv,
        GATConv,
        GATv2Conv,
        GINConv,
        SGConv,
        SAGEConv,
        TransformerConv,
    )
    if encoder_type == 'gat':
        return GATConv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    elif encoder_type == 'gatv2':
        return GATv2Conv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    elif encoder_type == 'gcn':
        return GCNConv(in_channels=in_channels, out_channels=out_channels)
    elif encoder_type == "gin":
        return GINConv(Linear(in_channels=in_channels, out_channels=out_channels), train_eps=True)
    elif encoder_type == "sgc":
        return SGConv(in_channels=in_channels, out_channels=out_channels)
    elif encoder_type == "sage":
        return SAGEConv(in_channels=in_channels, out_channels=out_channels)
    elif encoder_type == 'transformer':
        return TransformerConv(in_channels=-1, out_channels=out_channels, heads=num_heads)
    else:
        raise ValueError('Unexplored')


def to_sparse_tensor(edge_index, num_nodes):
    from torch_sparse import SparseTensor
    return SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to(edge_index.device)


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


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


"""
Net for Inductive Cognitive Diagnosis Model
"""
import dgl
from dgl.base import DGLError
from dgl.nn.pytorch import GATConv, GATv2Conv
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair
class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                                             : graph.num_dst_nodes()
                                             ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = h_neigh
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class Weighted_Summation(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Weighted_Summation, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
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


class SAGENet(nn.Module):
    def __init__(self, dim, layers_num=2, type='mean', device='cpu', drop=True, d_1=0.05, d_2=0.1):
        super(SAGENet, self).__init__()
        self.drop = drop
        self.type = type
        self.d_1 = d_1
        self.d_2 = d_2
        self.layers = []
        for i in range(layers_num):
            if type == 'mean' or type == 'pool':
                self.layers.append(SAGEConv(in_feats=dim, out_feats=dim, aggregator_type=type).to(device))
            elif type == 'gat':
                self.layers.append(GATConv(in_feats=dim, out_feats=dim, num_heads=4).to(device))
            elif type == 'gatv2':
                self.layers.append(GATv2Conv(in_feats=dim, out_feats=dim, num_heads=4).to(device))

    def forward(self, g, h):
        outs = [h]
        tmp = h
        from dgl import DropEdge
        for index, layer in enumerate(self.layers):
            drop = DropEdge(p=self.d_1 + self.d_2 * index)
            if self.drop:
                if self.training:
                    g = drop(g)
                if self.type != 'mean' and self.type != 'pool':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            else:
                if self.type != 'mean' and self.type != 'pool':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            outs.append(tmp / (1 + index))
        res = torch.sum(torch.stack(
            outs, dim=1), dim=1)
        return res