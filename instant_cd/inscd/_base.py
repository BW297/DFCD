import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from abc import abstractmethod
from sklearn.metrics import roc_auc_score, accuracy_score

from . import listener
from . import ruler
from . import unifier


class _Extractor:
    @abstractmethod
    def extract(self, **kwargs):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...


class _InteractionFunction:
    @abstractmethod
    def compute(self, **kwargs):
        ...

    @abstractmethod
    def transform(self, mastery, knowledge):
        ...

    def monotonicity(self):
        ...


class _CognitiveDiagnosisModel:
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, save_flag=False):
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.save_flag = save_flag
        # ellipsis members
        self.method = ...
        self.device: str = ...
        self.inter_func: _InteractionFunction = ...
        self.extractor: _Extractor = ...
        self.mastery_list: list = []
        self.diff_list: list = []
        self.knowledge_list: list = []

    def _train(self, datahub, set_type="train",
               valid_set_type=None, valid_metrics=None, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        unifier.train(datahub, set_type, self.extractor, self.inter_func, **kwargs)
        if valid_set_type is not None:
            self.score(datahub, valid_set_type, valid_metrics, **kwargs)
        if self.save:
            self.mastery_list.append(self.get_attribute('mastery'))
            self.diff_list.append(self.get_attribute('diff'))
            self.knowledge_list.append(self.get_attribute('knowledge'))

    def _predict(self, datahub, set_type: str, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return unifier.predict(datahub, set_type, self.extractor, self.inter_func, **kwargs)

    @listener
    def _score(self, datahub, set_type: str, metrics: list, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        pred_r = unifier.predict(datahub, set_type, self.extractor, self.inter_func, **kwargs)
        return ruler(self, datahub, set_type, pred_r, metrics)

    @abstractmethod
    def build(self, *args, **kwargs):
        ...

    @abstractmethod
    def train(self, datahub, set_type, valid_set_type=None, valid_metrics=None, **kwargs):
        ...

    @abstractmethod
    def predict(self, datahub, set_type, **kwargs):
        ...

    @abstractmethod
    def score(self, datahub, set_type, metrics: list, **kwargs) -> dict:
        ...

    @abstractmethod
    def diagnose(self):
        ...

    @abstractmethod
    def load(self, ex_path: str, if_path: str):
        ...

    @abstractmethod
    def save(self, ex_path: str, if_path: str):
        ...

    @abstractmethod
    def get_attribute(self, attribute_name):
        ...

    def cat_get_pred(self, datahub):
        adaptest_data = datahub['CAT']
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        num_concepts = adaptest_data.num_concepts
        pred_all = {}
        real_all = {}

        self.extractor.eval()
        self.inter_func.eval()

        with torch.no_grad():
            for sid, exercises in data.items():
                student_ids = [sid] * len(exercises)
                exercise_ids = list(exercises.keys())
                concepts_embs = [[float(concept in concept_map[qid]) for concept in range(num_concepts)] for qid in
                                 exercise_ids]

                student_ids_tensor = torch.LongTensor(student_ids).to(self.device)
                exercise_ids_tensor = torch.LongTensor(exercise_ids).to(self.device)
                concepts_embs_tensor = torch.Tensor(concepts_embs).to(self.device)
                _ = self.extractor.extract(student_ids_tensor, exercise_ids_tensor, concepts_embs_tensor)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                output = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_embs_tensor,
                    knowledge_ts=knowledge_ts
                ).detach().cpu().numpy().tolist()

                pred_all[sid] = dict(zip(exercise_ids, output))
                real_all[sid] = {qid: data[sid][qid] for qid in exercise_ids}

        self.extractor.train()
        self.inter_func.train()

        return pred_all, real_all

    def cat_evaluate(self, datahub):
        concept_map = datahub.get_concept_map()
        pred_all, real_all = self.cat_get_pred(datahub)
        real, pred = [], []
        for sid in pred_all:
            real.extend(real_all[sid].values())
            pred.extend(pred_all[sid].values())
        pred, real = np.array(pred), np.array(real)
        real = np.where(real < 0.5, 0, 1)
        auc = roc_auc_score(real, pred)
        acc = accuracy_score(real, (pred >= 0.5).astype(int))
        coverages = [
            len(set().union(*(concept_map[qid] for qid in datahub['CAT'].tested[sid]))) /
            len(set().union(*(concept_map[qid] for qid in datahub['CAT'].data[sid])))
            for sid in datahub['CAT'].data
        ]
        cov = np.mean(coverages)
        return {'auc': auc, 'cov': cov, 'acc': acc}

    def cat_adaptest_save(self, paths):
        if len(paths) != 2:
            raise ValueError("Expected two paths for saving the state dicts of extractor and inter_func.")

        torch.save(self.extractor.state_dict(), paths[0])
        torch.save(self.inter_func.state_dict(), paths[1])

    def cat_adaptest_load(self, paths):
        if len(paths) != 2:
            raise ValueError("Expected two paths for loading the state dicts of extractor and inter_func.")

        extractor_state_dict = torch.load(paths[0], map_location=self.device)
        self.extractor.load_state_dict(extractor_state_dict, strict=False)
        self.extractor.to(self.device)

        interaction_state_dict = torch.load(paths[1], map_location=self.device)
        self.inter_func.load_state_dict(interaction_state_dict, strict=False)
        self.inter_func.to(self.device)

    @staticmethod
    def cat_get_BE_weights(pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11 = torch.tensor(Pre_true[i], requires_grad=True)
                tensor_12 = torch.tensor(Pre_true[j], requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                import math
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i] * Pre_false[j] * diff_norm_00 + Pre_false[i] * Pre_true[j] * diff_norm_01 + \
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + Pre_true[i] * Pre_true[j] * diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    @staticmethod
    def cat_F_s_func(S_set, w_ij_matrix):
        return sum(max(w_ij_matrix[w_i][j] for j in S_set) for w_i in w_ij_matrix if w_i not in S_set)

    def cat_adaptest_update(self, datahub, update_config: dict):
        epoch = update_config.get('epoch', 10)
        lr = update_config.get('lr', 5e-4)
        weight_decay = update_config.get('weight_decay', 0)
        batch_size = update_config.get('batch_size', 256)
        adaptest_data = datahub['CAT']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([
            {'params': self.extractor.parameters(), 'lr': lr, "weight_decay": weight_decay},
            {'params': self.inter_func.parameters(), 'lr': lr, "weight_decay": weight_decay}
        ])
        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for epoch_i in tqdm(range(epoch), desc='updating model'):
            total_loss = 0.0
            for student_ids, question_ids, concepts_emb, labels in dataloader:
                student_ids, question_ids, concepts_emb, labels = (student_ids.to(self.device),
                                                                   question_ids.to(self.device),
                                                                   concepts_emb.to(self.device),
                                                                   labels.to(self.device).float())

                student_ts, diff_ts, disc_ts, knowledge_ts = self.extractor.extract(student_ids, question_ids,
                                                                                    concepts_emb)[:4]
                predictions = self.inter_func.compute(student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts,
                                                      q_mask=concepts_emb, knowledge_ts=knowledge_ts)

                loss = loss_func(predictions, labels.double())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.inter_func.monotonicity()
                total_loss += loss.item()


    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """
        sampled_set = set(sampled_elements)
        sampled_set.update(S_set)
        if question_id not in sampled_set:
            sampled_set.add(question_id)
        sampled_dict = {key: pred_all[key] for key in sampled_set}
        w_ij_matrix = self.cat_get_BE_weights(sampled_dict)
        S_set = set(S_set)
        Sp_set = S_set.union({question_id})
        return self.cat_F_s_func(Sp_set, w_ij_matrix) - self.cat_F_s_func(S_set, w_ij_matrix)

    def expected_model_change(self, sid: int, qid: int, datahub, pred_all: dict, update_config: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epoch = update_config.get('epoch', 10)
        lr = update_config.get('lr', 5e-4)
        weight_decay = update_config.get('weight_decay', 0)
        adaptest_data = datahub['CAT']
        optimizer = optim.Adam(self.extractor.parameters(), lr=lr, weight_decay=weight_decay)
        original_state_dict = copy.deepcopy(self.extractor.state_dict())
        init_mastery = self.diagnose()
        student_ids = torch.LongTensor([sid]).to(self.device)
        question_ids = torch.LongTensor([qid]).to(self.device)
        concepts_embs = torch.Tensor(
            [[1.0 if i in adaptest_data.concept_map[qid] else 0.0 for i in range(adaptest_data.num_concepts)]]).to(
            self.device)
        labels = [torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device)]
        weight_changes = []
        for label in labels:
            self.extractor.load_state_dict(original_state_dict)
            for ep in range(epoch):
                optimizer.zero_grad()
                student_ts, diff_ts, disc_ts, knowledge_ts = self.extractor.extract(student_ids, question_ids,
                                                                                       concepts_embs)[:4]
                pred = self.inter_func.compute(student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts,
                                               q_mask=concepts_embs, knowledge_ts=knowledge_ts)
                loss = nn.BCELoss()(pred, label.double())
                loss.backward()
                optimizer.step()

            updated_weights = self.diagnose().data.clone()
            weight_changes.append(torch.norm(updated_weights -init_mastery).item())
        self.extractor.load_state_dict(original_state_dict)
        pred = pred_all[sid][qid]
        return pred * weight_changes[0] + (1 - pred) * weight_changes[1]
