import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
class SCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.device = device
        self.dtype = dtype

        self.prednet_full1 = nn.Linear(self.knowledge_num, self.knowledge_num, bias=False, dtype=dtype).to(self.device)

        self.prednet_full2 = nn.Linear(self.knowledge_num, self.knowledge_num, bias=False, dtype=dtype).to(self.device)

        self.prednet_full3 = nn.Linear(self.knowledge_num, self.knowledge_num, dtype=dtype).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]
        preference = torch.sigmoid(self.prednet_full1(student_ts))
        diff = torch.sigmoid(self.prednet_full2(diff_ts))
        o = torch.sigmoid(self.prednet_full3(preference - diff))
        sum_out = torch.sum(o * q_mask, dim=1)
        count_of_concept = torch.sum(q_mask, dim=1)
        output = sum_out / count_of_concept
        output = output.unsqueeze(1)
        return output.view(-1)

    def transform(self, mastery, knowledge):
        preference = torch.sigmoid(self.prednet_full1(mastery))
        o = torch.sigmoid(self.prednet_full3(preference))
        return o

    def monotonicity(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
