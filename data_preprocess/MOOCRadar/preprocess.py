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
    stu_question_map = {}
    for line in open(f'MOOCRadar/data/action/student-problem.json', 'r'):
        line = json.loads(line)
        stu_question_map[line['user_id']] = []
        for log in line['seq']:
            stu_question_map[log['user_id']].append((log['problem_id'], log['is_correct']))
    
    question_info_map = {}
    concept_set = set()
    for line in open(f'MOOCRadar/data/entity/problem.json', 'r', errors='ignore', encoding='utf-8'):
        line = json.loads(line)
        question_info_map[line['problem_id']] = {}
        #print(json.dumps(eval(line['detail'])['option'], ensure_ascii=False).encode('utf8').decode())
        question_info_map[line['problem_id']]['content'] = eval(line['detail'])['content'] + json.dumps(eval(line['detail'])['option'], ensure_ascii=False).encode('utf8').decode()
        question_info_map[line['problem_id']]['concepts'] = line['concepts']
        for concept in line['concepts']:
            concept_set.add(concept)
    
    concepts_map = {}
    questions_map = {}
    concepts_questions = {}
    questions_concepts = {}
    for uid in tqdm(stu_question_map):
        data = stu_question_map[uid]
        for log in data:
            question = log[0]
            for concept in question_info_map[question]['concepts']:
                if concepts_map.get(concept) is None:
                    concepts_map[concept] = 1
                else:
                    concepts_map[concept] += 1
            if questions_map.get(question) is None:
                questions_map[question] = 1
            else:
                questions_map[question] += 1
            for concept in question_info_map[question]['concepts']:
                if concepts_questions.get(concept) is None:
                    concepts_questions[concept] = []
                if question not in concepts_questions[concept]:
                    concepts_questions[concept].append(question)
    
    sorted_concepts_map_by_value_desc = sorted(concepts_map.items(), key=lambda x: x[1], reverse=True)
    sorted_questions_map_by_value_desc = sorted(questions_map.items(), key=lambda x: x[1], reverse=True)
    candidate_questions = []
    for _ in sorted_concepts_map_by_value_desc[:config['know_num']]:
        for question in concepts_questions[_[0]]:
            candidate_questions.append(question)
    candidate_questions = random.sample(candidate_questions, k=config['exer_num'])
    student_map = {}
    for uid in tqdm(stu_question_map):
        data = stu_question_map[uid]
        if student_map.get(uid) is None:
            student_map[uid] = 0
        for log in data:
            question = log[0]
            if question in candidate_questions:
                student_map[uid] += 1
    
    sorted_students_map_by_value_desc = sorted(student_map.items(), key=lambda x: x[1], reverse=True)
    stu_num = []
    for _ in sorted_students_map_by_value_desc[:config['stu_num']]:
        stu_num.append(_[0])

    response_data = dict()
    stu_response_data = dict() 
    least_respone_num = config['least_respone_num']

    for stu in tqdm(stu_num, desc='Filter student'):
        for data in stu_question_map[stu]:
            uid, question, response = stu, data[0], data[1]
                # if questions[i] in candidate_questions:
            tmp_data = (uid, question)
            response_data[tmp_data] = response

    for key, value in tqdm(response_data.items(), desc='Filter the least respone number'):
        if key[0] not in stu_response_data:
            stu_response_data[key[0]] = []
        stu_response_data[key[0]].append([key[0], key[1], int(value)])
    
    TotalData = []
    stu_map = dict()
    cnt_stu = 0
    question_set = set()
    cnt_question = 0
    question_map = dict()
    cnt_concept = 0
    concept_map = dict()
    reverse_question_map = {}
    concept_set = set()

    for key, value in tqdm(stu_response_data.items(), desc='Remap student_id, question_id and concept_id'):
        # print(value)
        if len(value) >= least_respone_num:
            # print(value)
            stu_map[key] = cnt_stu
            cnt_stu += 1
            for data in value:
                # print(data[1])
                question_set.add(data[1])
                for concept in question_info_map[data[1]]['concepts']:
                    # print(concept)
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
                # concept = q_data[str(data[1])]['kc_routes'][0].split('----')[-1]
                for concept in question_info_map[data[1]]['concepts']:
                    q_matrix[question_map[data[1]]][concept_map[concept]] = 1
    
    print('Final student number: {}, Final question number: {}, Final concept number: {}, Final response number: {}'.format(cnt_stu, cnt_question, cnt_concept, len(TotalData)))
    np.savetxt('MOOCRadar/result/data/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('MOOCRadar/result/data/q.csv', q_matrix, delimiter=',')
    np.savetxt('../data/MOOCRadar/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('../data/MOOCRadar/q.csv', q_matrix, delimiter=',')
    config_map = {}
    config_map['stu_map'] = stu_map
    config_map['question_map'] = question_map
    config_map['concept_map'] = concept_map
    config_map['reverse_question_map'] = reverse_question_map
    with open('MOOCRadar/result/data/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)
    with open('../data/MOOCRadar/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)
