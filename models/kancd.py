import torch
import numpy as np
import torch.nn as nn
from base import BaseModel
from decoders import Positive_MLP
from utils import NoneNegClipper

class KaNCD(BaseModel):
    def __init__(self, config):
        super(KaNCD, self).__init__(config)
        if config['exp_type'] == 'normal':
            self.student_emb = nn.Embedding(config['stu_num'], config['dim']).to(self.device)
            self.exercise_emb = nn.Embedding(config['prob_num'], config['dim']).to(self.device)
            self.knowledge_emb = nn.Parameter(torch.zeros(config['know_num'], config['dim'])).to(self.device)
        elif config['exp_type'] == 'pre':
            total_emb = self.config['train_data'].x_llm
            self.stu_total_emb = total_emb[:config['stu_num'],]
            self.exer_total_emb = total_emb[config['stu_num']:config['stu_num']+config['prob_num'],]
            self.know_total_emb = total_emb[config['stu_num']+config['prob_num']:,]
            self.student_emb = nn.Embedding(config['stu_num'], config['dim']).to(self.device)
            self.exercise_emb = nn.Linear(config['in_channels_llm'], config['dim']).to(self.device)
            self.knowledge_emb = nn.Parameter(torch.zeros(config['know_num'], config['dim'])).to(self.device)

        self.e_discrimination = nn.Embedding(config['prob_num'], 1).to(self.device)
        self.k_diff_full = nn.Linear(config['dim'], 1).to(self.device)
        self.stat_full = nn.Linear(config['dim'], 1).to(self.device)
        self.positive_mlp = Positive_MLP(config).to(self.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        batch = student_id.shape[0]
        if self.config['exp_type'] == 'normal':
            if self.training or self.config['split'] == 'Original':
                stu_emb = self.student_emb(student_id)
                exer_emb = self.exercise_emb(exercise_id)
            elif self.config['split'] == 'Stu':
                if self.config['embedding_method'] == 'Mean':
                    stu_emb = torch.mean(
                        self.student_emb(torch.Tensor(self.config['exist_idx']).to(torch.int).to(self.device)), dim=0)
                    stu_emb = stu_emb.unsqueeze(0).expand(batch, -1)
                elif self.config['embedding_method'] == 'Nearest':
                    stu_emb = self.student_emb(
                        torch.Tensor(self.config['nearest'][student_id.cpu()].reshape(batch, )).to(
                            torch.int).to(self.device))
                exer_emb = self.exercise_emb(exercise_id)
            elif self.config['split'] == 'Exer' or self.config['split'] == 'Know':
                if self.config['embedding_method'] == 'Mean':
                    exer_emb = torch.mean(
                        self.exercise_emb(torch.Tensor(self.config['exist_idx']).to(torch.int).to(self.device)), dim=0)
                    exer_emb = exer_emb.unsqueeze(0).expand(batch, -1)

                elif self.config['embedding_method'] == 'Nearest':
                    exer_emb = self.exercise_emb(
                        torch.Tensor(self.config['nearest'][exercise_id.cpu()].reshape(batch, )).to(
                            torch.int).to(self.device))
                stu_emb = self.student_emb(student_id)
        elif self.config['exp_type'] == 'pre':
            stu_emb = self.student_emb(student_id)
            exer_emb = self.exercise_emb(self.exer_total_emb[exercise_id])

        stu_emb = stu_emb.view(batch, 1, self.config['dim']).repeat(1, self.config['know_num'], 1)
        exer_emb = exer_emb.view(batch, 1, self.config['dim']).repeat(1, self.config['know_num'], 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.config['know_num'], -1)

        k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        e_discrimination = torch.sigmoid(self.e_discrimination(exercise_id))

        state = knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        return self.positive_mlp.forward(state).view(-1)

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.positive_mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

    def get_mastery_level(self):
        with torch.no_grad():
            blocks = torch.split(torch.arange(self.config['stu_num']).to(device=self.device), 5)
            mas = []
            for block in blocks:
                batch = block.shape[0]
                if self.config['exp_type'] == 'pre':
                    # stu_emb = self.student_emb(self.stu_total_emb[block.cpu()])
                    stu_emb = self.student_emb(block)
                elif self.config['split'] == 'Stu':
                    if self.config['embedding_method'] == 'Mean':
                        stu_emb = torch.mean(self.student_emb(torch.Tensor(self.config['exist_idx']).to(torch.int).to(self.device)), dim=0)
                        stu_emb = stu_emb.unsqueeze(0).expand(batch, -1)
                    elif self.config['embedding_method'] == 'Nearest':
                        stu_emb = torch.Tensor(self.config['nearest'][block.cpu()].reshape(batch, )).to(torch.int).to(self.device)
                        stu_emb = self.student_emb(stu_emb)
                else:
                    stu_emb = self.student_emb(block)
                batch, dim = stu_emb.size()
                stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.config['know_num'], 1)
                knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.config['know_num'], -1)
                stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
                mas.append(stat_emb.detach().cpu().numpy())
        return np.vstack(mas)