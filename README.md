# DFCD</u>-NeurIPS 2024

We provide comprehensive instructions on how to run DFCD in the ***<u>"exps"</u>*** directory. 

### Noting: Although the size of raw datasets are relatively small, but in our paper, we include the text semantic features in the framework which need a huge space to be stored after embedding. And such a huge size of the embeddings is not supported to be uploaded on the github, so you need to process datasets by yourself. But we have tried our best to provide you with the details on how to process the raw dataset, you can run the data preprocess in the ***<u>"data_preprocess"</u>*** directory with the following instruction.

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
Please install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```

# Dataset
You can download three datasets used in our paper here, but the raw dataset which is needed in this paper has been included.

NeurIPS2020: https://eedi.com/projects/neurips-education-challenge

XES3G5M: https://github.com/ai4ed/XES3G5M

MOOCRadar: https://github.com/THU-KEG/MOOC-Radar

# Data Preprocess

You should process datasets by yourself, you need first 

> cd data_preprocess

Noting: We have include the raw dataset in our folder with name of the datasets, but you need to unzip the dataset first, please make sure that you unzip the  `data.zip` in every dataset folder

Then, you can process different dataset with different settings using following command example:

```shell
python main_filter.py --dataset XES3G5M --seed 0 --stu_num 2000 --exer_num 2000 --know_num 200 --least_respone_num 50

python main_embedding.py --dataset XES3G5M --llm BAAI
```

But if you want to use dataset setting in our paper, just run the run.sh using following command: 

```shell
bash run.sh
```

Noting: The processing need the OpenAI api keys, please using the following command to export your OpenAI api keys:

```shell
export OPENAI_API_KEY=<Your OpenAI API key>
```



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

### Standard Scenario 

```shell
python dfcd_exp.py --method=dfcd --data_type=XES3G5M --lr=1e-4 --test_size=0.2 --seed=0 --batch_size=1024 --device=cuda:0 --epoch=20 --encoder_type=transformer --split=Original --mode=2 --text_embedding_model=openai
```

If you want to change the dataset, just replace the parameter of `--data_type` with the name of your target dataset such as NeurIPS2020 or MOOCRadar

Noting: If you don't use the setting in our paper, please make sure that you change the `config.json` in every dataset folder and the `data_params_dict.py` in data folder with your setting. 
