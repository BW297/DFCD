import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
from tqdm import tqdm

class MAATStrategy(AbstractStrategy):
    """
    'Model Agnostic Adaptive Testing'
    """

    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'MAAT'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, datahub, **kwargs):
        config = kwargs.get('update_config')
        student_list = kwargs.get('student_list')
        pred_all, _ = model.cat_get_pred(datahub)
        selection = {}
        for sid in tqdm(student_list, desc='selecting'):
            untested_questions = np.array(list(datahub['CAT'].untested[sid]))
            emc_arr = [model.expected_model_change(sid, qid, datahub, pred_all, config) for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, datahub['CAT']))
        return selection