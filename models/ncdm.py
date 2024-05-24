import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from decoders import Positive_MLP
from utils import NoneNegClipper


class NCDM(BaseModel):

    def __init__(self, config):
        self.knowledge_dim = config['know_num']
        self.exer_n = config['prob_num']
        self.emb_num = config['stu_num']
        self.stu_dim = self.knowledge_dim
        super(NCDM, self).__init__(config)

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim).to(self.device)
        if self.config['exp_type'] == 'normal':
            self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim).to(self.device)
            self.e_difficulty = nn.Embedding(self.exer_n, 1).to(self.device)
        elif self.config['exp_type'] == 'pre':
            total_emb = self.config['train_data'].x_llm
            self.exer_total_emb = total_emb[config['stu_num']:config['stu_num']+config['prob_num'],]
            self.k_difficulty = nn.Linear(self.config['in_channels_llm'], self.knowledge_dim).to(self.device)
            self.e_difficulty = nn.Linear(self.config['in_channels_llm'], 1).to(self.device)
        self.positive_mlp = Positive_MLP(config).to(self.device)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point, mode='train'):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        if self.config['exp_type'] == 'normal':
            k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
            e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        elif self.config['exp_type'] == 'pre':
            k_difficulty = torch.sigmoid(self.k_difficulty(self.exer_total_emb[input_exercise]))
            e_difficulty = torch.sigmoid(self.e_difficulty(self.exer_total_emb[input_exercise]))  # * 10
        # prednet
        state = e_difficulty * (stat_emb) * input_knowledge_point
        return self.positive_mlp.forward(state).view(-1)
    
    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.positive_mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

    def get_mastery_level(self):
        return torch.sigmoid(self.student_emb.weight.detach().cpu()).numpy()

