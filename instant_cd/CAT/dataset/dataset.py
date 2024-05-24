from collections import defaultdict, deque


class Dataset(object):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        self._raw_data = data
        self._concept_map = concept_map
        self.__num_students = num_students
        self.__num_questions = num_questions
        self.__num_concepts = num_concepts
        
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

    @property
    def num_students(self):
        return self.__num_students

    @property
    def num_questions(self):
        return self.__num_questions

    @property
    def num_concepts(self):
        return self.__num_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map
