import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
from .._base import _Extractor

from torch.nn import Module, Sequential, Linear, ReLU, Dropout, LogSoftmax
import dgl
from dgl.nn.pytorch import RelGraphConv


class RGCN(nn.Module):
    def __init__(
            self,
            knowledge_num,
            num_rels,
            gcn_layers,
            regularizer="basis",
            num_bases=-1,
            dropout=0.0,
            ns_mode=False,
            device="cpu"
    ):
        super(RGCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.knowledge_num = knowledge_num
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = -1
        if num_bases == -1:
            self.num_bases = num_rels
        self.device = device

        self.dropout = dropout
        self.ns_mode = ns_mode
        self.rgcn_layers = self.get_rgcn_layer()

    def get_rgcn_layer(self):
        rgcn_layer = []
        for idx, i in enumerate(range(self.gcn_layers)):
            rgc = RelGraphConv(
                self.knowledge_num, self.knowledge_num, self.num_rels, self.regularizer, self.num_bases, self_loop=False
            ).to(self.device)
            rgcn_layer.append(rgc)
            if idx != self.gcn_layers - 1:
                rgcn_layer.append(nn.ReLU())
        return rgcn_layer

    def forward(self, g, feat, etype):
        for layer in self.rgcn_layers:
            if isinstance(layer, RelGraphConv):
                feat = layer(g, feat, etype, g.edata["norm"])
            else:
                feat = layer(feat)
        return feat
        # h = self.conv1(g, feat, etype)
        # h = self.dropout(F.relu(h))
        # h = self.conv2(g, h, etype)
        # return h


class RGCN_Extractor(_Extractor, nn.Module):
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

    # def get_graph_layer(self):
    #     return [RGCLayer(input_dim=self.knowledge_num, h_dim=self.knowledge_num, num_base=40, drop_prob=0.1,
    #                      device=self.device) for i in
    #             range(self.gcn_layers)]

    def get_graph(self, graph):
        self.graph = graph[0].to(self.device)
        self.etype = graph[1]
        self.rgcn = RGCN(self.knowledge_num, self.knowledge_num, gcn_layers=self.gcn_layers, device=self.device).to(self.device)

    def __common_forward(self):
        stu_emb = self.__student_emb(self.stu_index).to(self.device)
        exer_emb = self.__exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.__knowledge_emb(self.k_index).to(self.device)

        all_weight = torch.cat([stu_emb, exer_emb], dim=0).to(self.device)
        weight = self.rgcn.forward(self.graph, all_weight, torch.tensor(self.etype).to(self.device))
        stu_emb_1, exer_emb_1 = weight[:self.student_num], weight[self.student_num:]
        return stu_emb_1, exer_emb_1, kn_emb

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
