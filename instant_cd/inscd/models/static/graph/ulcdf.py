import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF, CDMFKC_IF, KSCD_IF, IRT_IF, SCD_IF
from ....extractor import ULCDF_Extractor


class ULCDF(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        ULCDF ...

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

    def build(self, latent_dim=32, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear',
              leaky=0.8, keep_prob=0.9,
              dtype=torch.float32, hidden_dims: list = None, activation='ELU', mode='all', **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.device = device
        self.mode = mode
        self.extractor = ULCDF_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            activation=activation,
            mode=mode,
        )
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

        if self.mode == 'Q':
            ek_graph = np.zeros(shape=datahub.q_matrix.shape)
        else:
            ek_graph = datahub.q_matrix.copy()

        se_graph_right, se_graph_wrong = [self.__create_adj_se(datahub[set_type], is_subgraph=True)[i] for i in
                                          range(2)]
        se_graph = self.__create_adj_se(datahub[set_type], is_subgraph=False)
        sk_graph = self.__create_adj_sk(datahub[set_type], datahub.q_matrix)
        graph_dict = {
            'right': self.__final_graph(se_graph_right, sk_graph, ek_graph),
            'wrong': self.__final_graph(se_graph_wrong, sk_graph, ek_graph),
            'all': self.__final_graph(se_graph, sk_graph, ek_graph)
        }
        self.extractor.get_graph_dict(graph_dict)
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

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_sk(self, np_train, q):
        sk_np = np.zeros(shape=(self.student_num, self.knowledge_num))
        if self.mode == 'I':
            return sk_np
        for k in range(np_train.shape[0]):
            stu_id = np_train[k, 0]
            exer_id = np_train[k, 1]
            skills = np.where(q[int(exer_id)] != 0)[0]
            sk_np[int(stu_id), skills] = 1
        return sk_np

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))
            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]
            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, sk, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num: se_num] = se
        tmp[:self.student_num, se_num:sek_num] = sk
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T
        graph = sp.csr_matrix(graph)
        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)
