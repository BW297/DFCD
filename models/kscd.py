import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel


class KSCD(BaseModel):

    def __init__(self, config):
        self.knowledge_n = config['know_num']
        self.exer_n = config['prob_num']
        self.student_n = config['stu_num']
        self.emb_dim = config['dim']
        self.prednet_input_len = self.knowledge_n
        super(KSCD, self).__init__(config)

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim).to(self.device)
        if self.config['exp_type'] == 'normal':
            self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim).to(self.device)
        elif self.config['exp_type'] == 'pre':
            total_emb = self.config['train_data'].x_llm
            self.exer_total_emb = total_emb[config['stu_num']:config['stu_num']+config['prob_num'],]
            self.exercise_emb = nn.Linear(self.config['in_channels_llm'], self.emb_dim).to(self.device)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim)).to(self.device)
        self.disc_mlp = nn.Linear(self.emb_dim, 1).to(self.device)
        self.f_sk = nn.Linear(self.knowledge_n + self.emb_dim, self.knowledge_n).to(self.device)
        self.f_ek = nn.Linear(self.knowledge_n + self.emb_dim, self.knowledge_n).to(self.device)
        self.f_se = nn.Linear(self.knowledge_n, 1).to(self.device)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point, mode='train'):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        if self.config['exp_type'] == 'normal':
            exer_emb = self.exercise_emb(input_exercise)
        elif self.config['exp_type'] == 'pre':
            exer_emb = self.exercise_emb(self.exer_total_emb[input_exercise])
        stu_ability = torch.sigmoid(stu_emb @ self.knowledge_emb.T)
        diff_emb = torch.sigmoid(exer_emb @ self.knowledge_emb.T)
        disc = torch.sigmoid(self.disc_mlp(exer_emb))
        batch, dim = stu_emb.size()
        stu_emb = stu_ability.unsqueeze(1).repeat(1, self.knowledge_n, 1)
        diff_emb = diff_emb.unsqueeze(1).repeat(1, self.knowledge_n, 1)
        Q_relevant = input_knowledge_point.unsqueeze(2).repeat(1, 1, self.knowledge_n)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        s_k_concat = torch.sigmoid(self.f_sk(torch.cat([stu_emb, knowledge_emb], dim=-1)))
        e_k_concat = torch.sigmoid(self.f_ek(torch.cat([diff_emb, knowledge_emb], dim=-1)))
        return torch.sigmoid(disc * self.f_se(torch.mean((s_k_concat - e_k_concat) * Q_relevant, dim=1))).view(-1)
    
    def monotonicity(self):
        pass

    def get_mastery_level(self):
        return torch.sigmoid(self.student_emb.weight @ self.knowledge_emb.T).detach().cpu().numpy()

