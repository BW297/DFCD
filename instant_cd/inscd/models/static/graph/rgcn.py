import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, RCD_IF, KANCD_IF
from ....extractor import RGCN_Extractor

import scipy.sparse as sp
from scipy.sparse import coo_matrix

class RGCN(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        RGCN ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(student_num, exercise_num, knowledge_num)

    def build(self, device: str = "cpu", if_type='ncd', hidden_dims: list = None,
              dtype=torch.float32, gcn_layers=3, dropout=0.1, **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        if if_type == 'kancd':
            latent_dim = 32
        else:
            latent_dim = self.knowledge_num

        self.extractor = RGCN_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            if_type=if_type,
            gcn_layers=gcn_layers
        )
        self.device = device

        if if_type == 'ncd':
            self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                     hidden_dims=hidden_dims,
                                     dropout=0,
                                     device=device,
                                     dtype=dtype)
        elif 'dp' in if_type:
            self.inter_func = DP_IF(knowledge_num=self.knowledge_num,
                                    hidden_dims=hidden_dims,
                                    dropout=0,
                                    device=device,
                                    dtype=dtype,
                                    kernel=if_type)
        elif 'rcd' in if_type:
            self.inter_func = RCD_IF(
                knowledge_num=self.knowledge_num,
                device=self.device,
                dtype=dtype
            )
        elif 'mirt' in if_type:
            self.inter_func = MIRT_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=16,
                device=device,
                dtype=dtype,
                utlize=True)

        elif 'kancd' in if_type:
            self.inter_func = KANCD_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
                hidden_dims=hidden_dims,
                dropout=0.5
            )

        else:
            raise ValueError("Remain to be aligned....")

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=0.0001, weight_decay=0.0005, batch_size=256):
        self.datahub = datahub
        self.extractor.get_graph(self.build_graph4SE())
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)

    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa", 'ap']
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def build_graph4SE(self):
        node = self.student_num + self.exercise_num
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        data = self.datahub['train']
        etype = []
        for index in range(data.shape[0]):
            stu_id = data[index, 0]
            exer_id = data[index, 1]
            edge_list.append((int(stu_id), int(exer_id + self.student_num)))
            edge_list.append((int(exer_id + self.student_num), int(stu_id)))

            if int(data[index, 2]) == 1:
                etype.extend([1, 1])
            else:
                etype.extend([0, 0])
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.edata["norm"] = dgl.norm_by_dst(g).unsqueeze(-1)
        return g, etype