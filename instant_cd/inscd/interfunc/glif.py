import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class GLIF_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, hidden_dims: list, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype),
                        'activation0': nn.Tanh()
                    }
                )
            else:
                layers.update(
                    {
                        'dropout{}'.format(idx): nn.Dropout(p=self.dropout),
                        'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype),
                        'activation{}'.format(idx): nn.Tanh()
                    }
                )
        layers.update(
            {
                'dropout{}'.format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                'linear{}'.format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1, dtype=self.dtype
                ),
                'activation{}'.format(len(self.hidden_dims)): nn.Sigmoid()
            }
        )

        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    @staticmethod
    def concept_distill(matrix, concept):
        coeff = 1.0 / torch.sum(matrix, dim=1)
        concept = matrix.to(torch.float64) @ concept
        concept_distill = concept * coeff[:, None]
        return concept_distill

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]

        exer_concept_distill = self.concept_distill(q_mask, knowledge_ts)
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts * exer_concept_distill) - torch.sigmoid(diff_ts * exer_concept_distill)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
