<<<<<<< HEAD
# DFCD</u>-NeurIPS 2024

We provide comprehensive instructions on how to run DFCD in the ***<u>"exps"</u>*** directory. And due to the memory limit, we unable to commit our data after processing directly. So you need to process datasets by yourself, you can run the data preprocess in the ***<u>"data_preprocess"</u>*** directory.

# Requirements	

```python
dgl==2.1.0+cu121
edustudio==1.1.1
FlagEmbedding==1.2.9
FlagEmbedding==1.2.10
InstructorEmbedding==1.0.1
InstructorEmbedding==1.0.1
joblib==1.3.2
langchain==0.2.1
langchain_core==0.2.1
langchain_google_genai==1.0.5
matplotlib==3.8.3
networkx==2.7
numpy==1.26.4
openai==1.30.2
pandas==2.2.2
protobuf==5.27.0
scikit_learn==1.4.1.post1
scipy==1.13.1
seaborn==0.13.2
torch==2.2.1
torch_geometric==2.5.3
torch_sparse==0.6.18+pt22cu121
tqdm==4.65.0
vegas==6.0.1
```


# Dataset
You can download three datasets used in our paper here.

NeurIPS2020: https://eedi.com/projects/neurips-education-challenge

XES3G5M: https://github.com/ai4ed/XES3G5M

MOOCRadar: https://github.com/THU-KEG/MOOC-Radar

# Data Preprocess

You should process datasets by yourself, you need first 

> cd data_preprocess

Then, you can process different dataset using following command example:

```shell
python main_filter.py --dataset XES3G5M --seed 0 --stu_num 2000 --exer_num 2000 --know_num 200 --least_respone_num 50

python main_embedding.py --dataset XES3G5M --llm BAAI
```

If you want process other dataset , just change the "--dataset" 


# Experiments

Firstly, you need

> cd exps

Then, you can run our framework in diffent scenario using following command:

### Unseen Student

```shell
python dfcd_exp.py --method=dfcd --data_type=XES3G5M --lr=1e-4 --test_size=0.2 --seed=0 --batch_size=1024 --device=cuda:0 --epoch=20 --encoder_type=transformer --split=Stu --mode=2 --text_embedding_model=openai
```
### Unseen Exercise

```shell
python dfcd_exp.py --method=dfcd --data_type=XES3G5M --lr=1e-4 --test_size=0.2 --seed=0 --batch_size=1024 --device=cuda:0 --epoch=20 --encoder_type=transformer --split=Exer --mode=2 --text_embedding_model=openai
```
### Unseen Concept

```shell
python dfcd_exp.py --method=dfcd --data_type=XES3G5M --lr=1e-4 --test_size=0.2 --seed=0 --batch_size=1024 --device=cuda:0 --epoch=20 --encoder_type=transformer --split=Know --mode=2 --text_embedding_model=openai
```

## Standard Scenario 

```shell
python dfcd_exp.py --method=dfcd --data_type=XES3G5M --lr=1e-4 --test_size=0.2 --seed=0 --batch_size=1024 --device=cuda:0 --epoch=20 --encoder_type=transformer --split=Standard --mode=2 --text_embedding_model=openai
```

