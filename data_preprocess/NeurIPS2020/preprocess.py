import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def fix_seeds(seed=101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
    np.random.seed(seed)

def run(config):
    fix_seeds(config['seed'])
    response_logs = pd.read_csv('NeurIPS2020/data/merged_data_V4.csv')
    questions = pd.read_excel("NeurIPS2020/data/data_refined.xlsx")

    response_logs = response_logs.loc[response_logs['QuestionId'].isin(questions['ID'].unique())]
    grouped = response_logs.groupby('UserId').size()
    grouped = grouped.loc[grouped > config['least_respone_num']]
    response_logs = response_logs.loc[response_logs['UserId'].isin(grouped.index)]

    original_stu_map = {}
    cnt = 0
    for stu in response_logs["UserId"].unique():
        original_stu_map[cnt] = stu
        cnt += 1
    
    response_data = dict()
    stu_response_data = dict() 
    stu_num = np.random.choice(np.arange(len(response_logs["UserId"].unique())), size=config['stu_num'], replace=False)
    least_respone_num = 0

    for stu in tqdm(stu_num, desc='Filter student'):
        for data in response_logs.loc[response_logs['UserId'] == original_stu_map[stu]].values:
            uid, question, response, concept = data[1], data[0], data[3], data[7]
            tmp_data = (uid, question, concept)
            response_data[tmp_data] = response
        

    for key, value in tqdm(response_data.items(), desc='Filter the least respone number'):
        if key[0] not in stu_response_data:
            stu_response_data[key[0]] = []
        stu_response_data[key[0]].append([int(key[0]), int(key[1]), int(value), key[2]])
    
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
                concept = data[3]
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
                concept = data[3]
                q_matrix[question_map[data[1]]][concept_map[concept]] = 1
    
    print('Final student number: {}, Final question number: {}, Final concept number: {}, Final response number: {}'.format(cnt_stu, cnt_question, cnt_concept, len(TotalData)))
    np.savetxt('NeurIPS2020/result/data/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('NeurIPS2020/result/data/q.csv', q_matrix, delimiter=',')
    np.savetxt('../data/NeurIPS2020/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('../data/NeurIPS2020/q.csv', q_matrix, delimiter=',')
    config_map = {}
    config_map['stu_map'] = stu_map
    config_map['question_map'] = question_map
    config_map['concept_map'] = concept_map
    config_map['reverse_question_map'] = reverse_question_map
    with open('NeurIPS2020/result/data/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)
    with open('../data/NeurIPS2020/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)

