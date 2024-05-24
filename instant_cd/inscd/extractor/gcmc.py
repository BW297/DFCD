import torch
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
from torch.nn import init
from dgl.base import DGLError
import torch.nn.functional as F
from .._base import _Extractor
def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == "leaky":
            return nn.LeakyReLU(0.1)
        elif act == "relu":
            return nn.ReLU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "softsign":
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act
def to_etype_name(rating):
    return str(rating).replace(".", "_")
class GCMCGraphConv(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(
        self, in_feats, out_feats, weight=True, device=None, dropout_rate=0.0
    ):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats)).double().to(self.device)
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used
            cj = graph.srcdata["cj"]
            ci = graph.dstdata["ci"]
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if weight is not None:
                print()
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata["h"] = feat
            graph.update_all(
                fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
            )
            rst = graph.dstdata["h"]
            rst = rst * ci

        return rst


class GCMC_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, if_type='ncd'):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.if_type = if_type
        self.latent_dim = latent_dim

        self.right_graph = ...
        self.wrong_graph = ...

        self.__student_emb = nn.Embedding(self.student_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.student_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exercise_num))).to(self.device)

        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph_layer(self):
        return [GCMCGraphConv(in_feats=self.knowledge_num, out_feats=self.knowledge_num, device=self.device) for i in range(self.gcn_layers)]

    def get_graph(self, graph):
        self.right_graph = graph['right'].to(self.device)
        self.wrong_graph = graph['wrong'].to(self.device)
        self.gcmc_right = nn.Sequential(
            *self.get_graph_layer()
        ).to(self.device)
        self.gcmc_wrong = nn.Sequential(
            *self.get_graph_layer()
        ).to(self.device)

    def graph_forward(self, graph, feat, graph_layers):
        for layer in graph_layers:
            feat = layer(graph, feat)
        return feat

    def __common_forward(self):
        stu_emb = self.__student_emb(self.stu_index).to(self.device)
        exer_emb = self.__exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.__knowledge_emb(self.k_index).to(self.device)

        all_weight = torch.cat([stu_emb, exer_emb], dim=0).to(self.device)
        weight_1 = self.graph_forward(self.right_graph, all_weight, self.gcmc_right)
        weight_2 = self.graph_forward(self.wrong_graph, all_weight, self.gcmc_wrong)
        stu_emb_1, exer_emb_1 = weight_1[:self.student_num], weight_1[self.student_num:]
        stu_emb_2, exer_emb_2 = weight_2[:self.student_num], weight_2[self.student_num:]
        return stu_emb_1 + stu_emb_2, exer_emb_1 + exer_emb_2, kn_emb

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward, knows_forward = self.__common_forward()
        batch_stu_emb = stu_forward[student_id]
        batch_exer_emb = exer_forward[exercise_id]
        disc_ts = self.__disc_emb(exercise_id)

        if self.if_type == 'rcd':
            batch_stu_ts = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])
            batch_exer_ts = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])
            knowledge_ts = knows_forward.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                   knows_forward.shape[0],
                                                                                   knows_forward.shape[1])
        else:
            batch_stu_ts = batch_stu_emb
            batch_exer_ts = batch_exer_emb
            knowledge_ts = knows_forward

        return batch_stu_ts, batch_exer_ts, disc_ts, knowledge_ts

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward, knows_forward = self.__common_forward()
        if self.if_type == 'rcd':
            student_ts = stu_forward.repeat(1, stu_forward.shape[1]).reshape(stu_forward.shape[0],
                                                                             stu_forward.shape[1],
                                                                             stu_forward.shape[1])

            # get batch exercise data
            diff_ts = exer_forward.repeat(1, exer_forward.shape[1]).reshape(exer_forward.shape[0],
                                                                            exer_forward.shape[1],
                                                                            exer_forward.shape[1])

            # get batch knowledge concept data
            knowledge_ts = knows_forward.repeat(stu_forward.shape[0], 1).reshape(stu_forward.shape[0],
                                                                                 knows_forward.shape[0],
                                                                                 knows_forward.shape[1])
        else:
            student_ts = stu_forward
            diff_ts = exer_forward
            knowledge_ts = knows_forward

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]

