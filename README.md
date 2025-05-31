# DFCD

The implementation for the KDD 2025 paper "A Dual-Fusion Cognitive Diagnosis Framework for Open Student Learning Environments". The paper are in the main.pdf , which are contained in the folder named paper.

# 💻 Requirements	

```python
FlagEmbedding==1.3.4
InstructorEmbedding==1.0.1
joblib==1.4.2
langchain==0.3.25
numpy==2.2.6
openai==1.82.0
pandas==2.2.3
scikit_learn==1.6.1
sentence_transformers==3.3.1
torch==2.4.0
torch_geometric==2.6.1
torch_sparse==0.6.18+pt24cu121
tqdm==4.66.5
```
Please install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```

# 🛠️ Data Preprocess

Although the size of raw datasets are relatively small, but in our paper, we include the text semantic features in the framework which need a huge space to be stored after embedding. And such a huge size of the embeddings is not supported to be uploaded on the github, so you need to process datasets by yourself. But we have tried our best to provide you with the details on how to process the raw dataset, you can run the data preprocess in the ***<u>"data_preprocess"</u>*** directory with the following instruction. You need first:

> cd data_preprocess

We have include the raw dataset in our folder with name of the datasets, but you need to unzip the dataset first, please make sure that you unzip the  `data.zip` in every dataset folder using following command:

```shell
unzip data.zip -d data
```

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



# 🛠️ Experiments

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

# Reference

Yuanhao Liu, Shuo Liu, Yimeng Liu, Chanjin Zheng, Wei Zhang, Hong Qian "A Dual-Fusion Cognitive Diagnosis Framework for Open Student Learning Environments" In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.

## Bibtex
```
@inproceedings{Liu2025DFCD,
 author = {Yuanhao Liu and Shuo Liu and Yimeng Liu and Chanjin Zheng and Wei Zhang and Hong Qian},
 booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
 title = {A Dual-Fusion Cognitive Diagnosis Framework for Open Student Learning Environments},
 year = {2025},
 address = {Toronto, Canada}
}
```
