python main_filter.py --dataset NeurIPS2020 --seed 0 --stu_num 2000 --exer_num 2000 --know_num 200 --least_respone_num 50
python main_filter.py --dataset XES3G5M --seed 0 --stu_num 2000 --exer_num 2000 --know_num 200 --least_respone_num 50
python main_filter.py --dataset MOOCRadar --seed 0 --stu_num 2000 --exer_num 1500 --know_num 500 --least_respone_num 50
python main_embedding.py --dataset XES3G5M --llm OpenAI
python main_embedding.py --dataset MOOCRadar --llm OpenAI
python main_embedding.py --dataset NeurIPS2020 --llm OpenAI
python main_embedding.py --dataset XES3G5M --llm BAAI
python main_embedding.py --dataset MOOCRadar --llm BAAI
python main_embedding.py --dataset NeurIPS2020 --llm BAAI
python main_embedding.py --dataset XES3G5M --llm m3e
python main_embedding.py --dataset MOOCRadar --llm m3e
python main_embedding.py --dataset NeurIPS2020 --llm m3e
python main_embedding.py --dataset XES3G5M --llm Instructor
python main_embedding.py --dataset MOOCRadar --llm Instructor
python main_embedding.py --dataset NeurIPS2020 --llm Instructor

# Final student number: 2000, Final question number: 454, Final concept number: 38, Final response number: 258233
# Final student number: 2000, Final question number: 1624, Final concept number: 241, Final response number: 207204
# Final student number: 2000, Final question number: 915, Final concept number: 696, Final response number: 385323