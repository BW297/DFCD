import torch
import torch.nn as nn
from collections import OrderedDict
from base import BaseModel
from decoders import Positive_MLP
from utils import NoneNegClipper


class IDCD(BaseModel):
    def __init__(self, config):
        super(IDCD, self).__init__(config)
        self.encoder_student = nn.Sequential(
            OrderedDict(
                [
                    ('f_layer_1', nn.Linear(self.config['prob_num'], 256)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', nn.Linear(256, config['know_num'])),
                    ('f_activate_2', nn.Sigmoid())
                ]
            )
        ).to(self.device)

        self.encoder_exercise = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(self.config['stu_num'], 512)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(512, 256)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(256, config['know_num'])),
                    ('g_activate_3', nn.Sigmoid())
                ]
            )
        ).to(self.device)
        self.positive_mlp = Positive_MLP(config).to(self.device)
        self.train_interaction_matrix = self.config['train_interaction_matrix']
        self.full_interaction_matrix = self.config['full_interaction_matrix']
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_x(self, mode='train'):
        if mode == 'train':
            student_factor = self.encoder_student(self.train_interaction_matrix)
            exercise_factor = self.encoder_exercise(self.train_interaction_matrix.T)
        else:
            student_factor = self.encoder_student(self.full_interaction_matrix)
            exercise_factor = self.encoder_exercise(self.full_interaction_matrix.T)
        return student_factor, exercise_factor

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        student_factor, exercise_factor = self.get_x(mode)
        stu_emb = student_factor[student_id]
        exer_emb = exercise_factor[exercise_id]
        state = knowledge_point * (torch.sigmoid(stu_emb) - torch.sigmoid(exer_emb))
        return self.positive_mlp.forward(state).view(-1)

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.positive_mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

    def get_mastery_level(self):
        if self.config['split'] == 'Stu' or self.config['split'] == 'Exer':
            mas = torch.sigmoid(self.encoder_student(self.config['full_interaction_matrix'])).detach().cpu().numpy()
        else:
            mas = torch.sigmoid(self.encoder_student(self.config['train_interaction_matrix'])).detach().cpu().numpy()
        return mas
