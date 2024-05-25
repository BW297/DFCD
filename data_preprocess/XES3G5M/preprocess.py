import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle
import random
import os

def fix_seeds(seed=101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
    np.random.seed(seed)

def run(config):
    fix_seeds(config['seed'])
    data_train = pd.read_csv('XES3G5M/data/kc_level/train_valid_sequences.csv')
    data_test = pd.read_csv('XES3G5M/data/kc_level/test.csv')

    with open(f'XES3G5M/data/metadata/questions.json', 'r') as file:
        q_data = json.load(file)

    with open(f'XES3G5M/data/metadata/kc_routes_map.json', 'r') as file:
        route_data = json.load(file)

    concepts_map = {}
    questions_map = {}
    concepts_questions = {}
    questions_concepts = {}
    for data in tqdm(data_train.values):
        uid, questions, concepts, responses, selectmasks = data[1], data[2].split(','), data[3].split(','), data[4].split(','), data[6].split(',')
        for concept in concepts:
            if concept == '-1':
                break
            if concepts_map.get(concept) is None:
                concepts_map[concept] = 1
            else:
                concepts_map[concept] += 1
        for question in questions:
            if question == '-1':
                break
            if questions_map.get(question) is None:
                questions_map[question] = 1
            else:
                questions_map[question] += 1
        for i in range(len(concepts)):
            if concepts[i] == '-1':
                break
            if concepts_questions.get(concepts[i]) is None:
                concepts_questions[concepts[i]] = []
            if questions[i] not in concepts_questions[concepts[i]]:
                concepts_questions[concepts[i]].append(questions[i])
            questions_concepts[questions[i]] = concepts[i]
        
    sorted_concepts_map_by_value_desc = sorted(concepts_map.items(), key=lambda x: x[1], reverse=True)
    sorted_questions_map_by_value_desc = sorted(questions_map.items(), key=lambda x: x[1], reverse=True)
    candidate_questions = []
    for _ in sorted_concepts_map_by_value_desc[:config['know_num']]:
        for question in concepts_questions[_[0]]:
            candidate_questions.append(question)
    candidate_questions = random.sample(candidate_questions, k=config['exer_num'])
    student_map = {}
    for data in tqdm(data_train.values):
        uid, questions, concepts, responses, selectmasks = data[1], data[2].split(','), data[3].split(','), data[4].split(','), data[6].split(',')
        student_map[uid] = 0
        for question in questions:
            if question == '-1':
                break
            if question in candidate_questions:
                student_map[uid] += 1
    sorted_students_map_by_value_desc = sorted(student_map.items(), key=lambda x: x[1], reverse=True)
    stu_num = []
    for _ in sorted_students_map_by_value_desc[:config['stu_num']]:
        stu_num.append(_[0])
    stu_num = np.array(stu_num)

    response_data = dict()
    stu_response_data = dict() 
    least_respone_num = config['least_respone_num']

    for stu in tqdm(stu_num, desc='Filter student'):
        for data in data_train.loc[data_train['uid'] == stu].values:
            uid, questions, responses, selectmasks = data[1], data[2].split(','), data[4].split(','), data[6].split(',')
            for i in range(len(questions)):
                if selectmasks[i] == '-1':
                    break
                if questions[i] in candidate_questions:
                    tmp_data = (uid, questions[i])
                    response_data[tmp_data] = responses[i]
        
        for data in data_test.loc[data_test['uid'] == stu].values:
            uid, questions, responses = data[1], data[2].split(','), data[4].split(',')
            for i in range(len(questions)):
                if questions[i] in candidate_questions:
                    tmp_data = (uid, questions[i])
                    response_data[tmp_data] = responses[i]

    for key, value in tqdm(response_data.items(), desc='Filter the least respone number'):
        if key[0] not in stu_response_data:
            stu_response_data[key[0]] = []
        stu_response_data[key[0]].append([int(key[0]), int(key[1]), int(value)])

    TotalData = []
    stu_map = dict()
    cnt_stu = 0
    question_set = set()
    cnt_question = 0
    question_map = dict()
    concept_set = set()
    cnt_concept = 0
    concept_map = dict()
    reverse_question_map = {}

    for key, value in tqdm(stu_response_data.items(), desc='Remap student_id, question_id and concept_id'):
        if len(value) >= least_respone_num:
            stu_map[key] = cnt_stu
            cnt_stu += 1
            for data in value:
                question_set.add(data[1])
                concept = q_data[str(data[1])]['kc_routes'][0].split('----')[-1]
                # for concept in q_data[str(data[1])]['kc_routes'][0].split('----'):
                concept_set.add(concept)

    for question in question_set:
        question_map[question] = cnt_question
        reverse_question_map[cnt_question] = question
        cnt_question += 1

    for concept in concept_set:
        concept_map[concept] = cnt_concept
        cnt_concept += 1
    
    TotalData = []
    q_matrix = np.zeros((cnt_question, cnt_concept))

    for key, value in tqdm(stu_response_data.items(), desc='Construct final data'):
        if len(value) >= least_respone_num:
            for data in value:
                TotalData.append([stu_map[data[0]], question_map[data[1]], data[2]])
                concept = q_data[str(data[1])]['kc_routes'][0].split('----')[-1]
                # for concept in q_data[str(data[1])]['kc_routes'][0].split('----'):
                q_matrix[question_map[data[1]]][concept_map[concept]] = 1
    

    print('Final student number: {}, Final question number: {}, Final concept number: {}, Final response number: {}'.format(cnt_stu, cnt_question, cnt_concept, len(TotalData)))
    np.savetxt('XES3G5M/result/data/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('XES3G5M/result/data/q.csv', q_matrix, delimiter=',')
    np.savetxt('../data/XES3G5M/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('../data/XES3G5M/q.csv', q_matrix, delimiter=',')
    config_map = {}
    config_map['stu_map'] = stu_map
    config_map['question_map'] = question_map
    config_map['concept_map'] = concept_map
    config_map['reverse_question_map'] = reverse_question_map
    with open('XES3G5M/result/data/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)
    with open('../data/XES3G5M/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)