import os
import openai
import pandas as pd
import json
from tqdm import tqdm
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

def generate_text(config, config_map, TotalData, q_data):
    student_prompt_template_right = "I was asked the question: {question}.\nAnd this question is about: {Name}.\n.And I give the correct answer."
    student_prompt_template_wrong = "I was asked the question: {question}.\nAnd this question is about: {Name}.\n.But I give the wrong answer."
    question_template = "The question's content is: {content} and it is about: {tag}."
    question_prompt = PromptTemplate.from_template(question_template)
    memory_prompt_right = PromptTemplate.from_template(student_prompt_template_right)
    memory_prompt_wrong = PromptTemplate.from_template(student_prompt_template_wrong)

    knowledge_original = []
    for concept in tqdm(config_map['concept_map']):
        knowledge_original.append(concept)
    config["knowledge_text"] = knowledge_original

    exercise_original = []
    for question in tqdm(config_map['question_map']):
        content = q_data[str(question)]['content']
        tag = q_data[str(question)]['kc_routes'][0]
        exercise_original.append(question_prompt.format_prompt(content=content, tag=tag).text)
    config["exercise_text"] = exercise_original

    student_original = []
    for student in tqdm(range(len(config_map['stu_map']))):
        tmp = []
        student_logs = TotalData.loc[TotalData['stu'] == student]
        for log in student_logs.values:
            question = q_data[str(config_map['reverse_question_map'][log[1]])]['content']
            Name = q_data[str(config_map['reverse_question_map'][log[1]])]['kc_routes'][0]
            if log[2] == 1:
                tmp.append(memory_prompt_right.format_prompt(question=question, Name=Name).text)
            else:
                tmp.append(memory_prompt_wrong.format_prompt(question=question, Name=Name).text)
        student_original.append(tmp)
    config['student_text'] = student_original  
            
    return config

def generate_embeddings_openai(config):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings()

    config["knowledge_embeddings"] = embeddings_model.embed_documents(config["knowledge_text"])
    config["exercise_embeddings"] = embeddings_model.embed_documents(config["exercise_text"])
    config["student_embeddings"] = []
    for student_text in tqdm(config['student_text']):
        config["student_embeddings"].append(embeddings_model.embed_documents(student_text))
    with open('../data/XES3G5M/embedding_openai.pkl', 'wb') as f:
        pickle.dump(config, f)
    with open('XES3G5M/result/embedding/embedding_openai.pkl', 'wb') as f:
        pickle.dump(config, f)

def generate_embeddings_BAAI(config):
    # with open('model/BAAI/bge-m3.pkl', 'rb') as file:
    #     embeddings_model = pickle.load(file)
    embeddings_model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)
    config["knowledge_embeddings"] = embeddings_model.encode(config["knowledge_text"], batch_size=12, max_length=8192,)['dense_vecs']
    config["exercise_embeddings"] = embeddings_model.encode(config["exercise_text"], batch_size=12, max_length=8192,)['dense_vecs']
    config["student_embeddings"] = []
    for student_text in tqdm(config['student_text']):
        config["student_embeddings"].append(embeddings_model.encode(student_text, batch_size=12, max_length=8192,)['dense_vecs'])
    with open('../data/XES3G5M/embedding_BAAI.pkl', 'wb') as f:
        pickle.dump(config, f)
    with open('XES3G5M/result/embedding/embedding_BAAI.pkl', 'wb') as f:
        pickle.dump(config, f)

def generate_embeddings_m3e(config):
    # with open('model/m3e/model.pkl', 'rb') as file:
    #     embeddings_model = pickle.load(file)
    embeddings_model = SentenceTransformer('moka-ai/m3e-base')
    config["knowledge_embeddings"] = embeddings_model.encode(config["knowledge_text"])
    config["exercise_embeddings"] = embeddings_model.encode(config["exercise_text"])
    config["student_embeddings"] = []
    for student_text in tqdm(config['student_text']):
        config["student_embeddings"].append(embeddings_model.encode(student_text))
    with open('../data/XES3G5M/embedding_m3e.pkl', 'wb') as f:
        pickle.dump(config, f)
    with open('XES3G5M/result/embedding/embedding_m3e.pkl', 'wb') as f:
        pickle.dump(config, f)

def generate_embeddings_Instructor(config):
    # with open('model/Instructor/model.pkl', 'rb') as file:
    #     embeddings_model = pickle.load(file)
    embeddings_model = INSTRUCTOR('hkunlp/instructor-base') 
    knowledge_text = []
    for text in config["knowledge_text"]:
        knowledge_text.append(['Represent the knowledge title:', text])
    config["knowledge_embeddings"] = embeddings_model.encode(knowledge_text)

    exercise_text = []    
    for text in config["exercise_text"]:
        exercise_text.append(['Represent the exercise description:', text])
    config["exercise_embeddings"] = embeddings_model.encode(exercise_text)

    config["student_embeddings"] = []
    for student_text in tqdm(config['student_text']):
        tmp = []
        for text in student_text:
            tmp.append(['Represent the student response log:', text])
        config["student_embeddings"].append(embeddings_model.encode(tmp))
    with open('../data/XES3G5M/embedding_instructor.pkl', 'wb') as f:
        pickle.dump(config, f)
    with open('XES3G5M/result/embedding/embedding_instructor.pkl', 'wb') as f:
        pickle.dump(config, f)


def run(arg):
    config = {}
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    with open('XES3G5M/result/data/map.pkl', 'rb') as f:
        config_map = pickle.load(f)
    TotalData = pd.read_csv("XES3G5M/result/data/TotalData.csv", header=None, names=['stu', 'exer', 'answervalue'])
    with open(f'XES3G5M/data/metadata/questions.json', 'r') as file:
        q_data = json.load(file)

    config = generate_text(config, config_map, TotalData, q_data)

    if arg['llm'] == 'OpenAI':
        generate_embeddings_openai(config)
    elif arg['llm'] == 'BAAI':
        generate_embeddings_BAAI(config)
    elif arg['llm'] == 'm3e':
        generate_embeddings_m3e(config)
    elif arg['llm'] == 'Instructor':
        generate_embeddings_Instructor(config)