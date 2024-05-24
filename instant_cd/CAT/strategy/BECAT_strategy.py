import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
import random
from tqdm import tqdm

class BECATstrategy(AbstractStrategy):
    """
    BECAT Strategy
    """
    
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BECAT'
    
    def adaptest_select(self, model: AbstractModel, datahub, **kwargs):
        S_sel_dict = kwargs['S_sel_dict']
        student_list = kwargs.get('student_list')
        pred_all, _ = model.cat_get_pred(datahub)
        selection = {}
        for sid in tqdm(student_list, desc='selecting'):
            tmplen = (len(S_sel_dict[sid]))
            untested_questions = np.array(list(datahub['CAT'].untested[sid]))
            sampled_elements = np.random.choice(untested_questions, tmplen + 5)
            untested_deltaq = [model.delta_q_S_t(qid, pred_all[sid],S_sel_dict[sid],sampled_elements) for qid in untested_questions]
            j = np.argmax(untested_deltaq)
            selection[sid] = untested_questions[j]
        return selection
    