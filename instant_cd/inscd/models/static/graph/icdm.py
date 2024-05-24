import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF, CDMFKC_IF, KSCD_IF, IRT_IF, SCD_IF, GLIF_IF
from ....extractor import ICDM_Extractor


class ICDM(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        ICDM ...

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

    def build(self, latent_dim=32, device: str = "cpu", gcn_layers: int = 3, if_type='ncd',
              dtype=torch.float32, hidden_dims: list = None, agg_type='mean', d_1=0.1, d_2=0.2, khop=3, **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        is_glif = if_type == "glif"

        self.device = device
        self.extractor = ICDM_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            agg_type=agg_type,
            khop=khop,
            d_1=d_1,
            d_2=d_2,
            is_glif=is_glif
        )


        if if_type == 'ncd':
            self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                     hidden_dims=hidden_dims,
                                     dropout=0,
                                     device=device,
                                     dtype=dtype)
        elif if_type == 'glif':
            self.inter_func = GLIF_IF(knowledge_num=self.knowledge_num,
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
        elif 'mirt' in if_type:
            self.inter_func = MIRT_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=32,
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
        elif 'cdmfkc' in if_type:
            self.inter_func = CDMFKC_IF(
                g_impact_a=0.5,
                g_impact_b=0.5,
                knowledge_num=self.knowledge_num,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'irt' in if_type:
            self.inter_func = IRT_IF(
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'kscd' in if_type:
            self.inter_func = KSCD_IF(
                dropout=0.5,
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype)
        elif 'scd' in if_type:
            self.inter_func = SCD_IF(
                knowledge_num=self.knowledge_num,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError("Remain to be aligned....")

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):
        self.datahub = datahub
        right, wrong = self.build_graph4SE()
        graph = {
            'right': right,
            'wrong': wrong,
            'Q': self.build_graph4CE(),
            'I': self.build_graph4SC()
        }
        self.extractor.get_graph_dict(graph)
        self.extractor.get_norm_adj(self.create_adj_mat())
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

    def build_graph4CE(self):
        node = self.exercise_num + self.knowledge_num
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        indices = np.where(self.datahub.q_matrix != 0)
        for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
            edge_list.append((int(know_id + self.exercise_num), int(exer_id)))
            edge_list.append((int(exer_id), int(know_id + self.exercise_num)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    def build_graph4SE(self):
        node = self.student_num + self.exercise_num
        g_right, g_wrong = dgl.DGLGraph(), dgl.DGLGraph()
        g_right.add_nodes(node)
        g_wrong.add_nodes(node)
        right_edge_list, wrong_edge_list = [], []
        data = self.datahub['train']
        for index in range(data.shape[0]):
            stu_id = data[index, 0]
            exer_id = data[index, 1]
            if int(data[index, 2]) == 1:
                right_edge_list.append((int(stu_id), int(exer_id + self.student_num)))
                right_edge_list.append((int(exer_id + self.student_num), int(stu_id)))
            else:
                wrong_edge_list.append((int(stu_id), int(exer_id + self.student_num)))
                wrong_edge_list.append((int(exer_id + self.student_num), int(stu_id)))
        right_src, right_dst = tuple(zip(*right_edge_list))
        wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
        g_right.add_edges(right_src, right_dst)
        g_wrong.add_edges(wrong_src, wrong_dst)
        return g_right, g_wrong

    def build_graph4SC(self):
        node = self.student_num + self.knowledge_num
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        sc_matrix = np.zeros(shape=(self.student_num, self.knowledge_num))
        data = self.datahub['train']
        for index in range(data.shape[0]):
            stu_id = data[index, 0]
            exer_id = data[index, 1]
            concepts = np.where(self.datahub.q_matrix[int(exer_id)] != 0)[0]
            for concept_id in concepts:
                if sc_matrix[int(stu_id), int(concept_id)] != 1:
                    edge_list.append((int(stu_id), int(concept_id + self.student_num)))
                    edge_list.append((int(concept_id + self.student_num), int(stu_id)))
                    sc_matrix[int(stu_id), int(concept_id)] = 1
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    @staticmethod
    def get_adj_matrix(tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    @staticmethod
    def sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()


    def create_adj_mat(self):
        n_nodes = self.student_num + self.exercise_num
        np_train = self.datahub['train']
        train_stu = np_train[:, 0]
        train_exer = np_train[:, 1]
        ratings = np.ones_like(train_stu, dtype=np.float64)
        tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.student_num)), shape=(n_nodes, n_nodes))
        return self.sp_mat_to_sp_tensor(self.get_adj_matrix(tmp_adj))
