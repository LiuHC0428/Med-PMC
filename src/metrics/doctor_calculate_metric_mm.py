import os
import re
import pdb
import json
import argparse
import numpy as np
import spacy
from rouge_score import rouge_scorer
from distinct_utils import distinct_n_corpus_level

nlp = spacy.load("en_core_web_sm")
scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

def remove_punctuation(text):
    # 使用正则表达式去除所有标点符号
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation

def calculate_metric(args):
    files = os.listdir(args.folder_path)
    files.sort() 
    pattern = r'\.json' 

    for file in files:
        if file.find('huatuo') != -1:
        # if file.endswith(r"\d+_multiple_6\.json"):
            if re.search(pattern, file):
                eval(os.path.join(args.folder_path, file),os.path.join(args.text_folder_path, file),os.path.join(args.no_mm_folder_path, file))
        
def eval(file_path, text_file_path, nomm_file_path):
    with open(file_path, "r") as f:
        datas = json.load(f)

    with open(text_file_path, "r") as f:
        text_datas = json.load(f)

    with open(nomm_file_path, "r") as f:
        nomm_datas = json.load(f)


    Score1 = []
    Score2 = []
    Score3 = []
    Score4 = []
    for i in range(len(datas)):
        if i == 6:
            continue
        try:
            data = datas[i]
            mm_token = False
            image_name = []
            Information_score = 0
            s1_dialog = []
            s3_dialog = []
            s4_report_info  = ''

            for j in range(len(data['history'])):
                dialog = data['history'][j]
                if dialog['state'] != 'D':
                    Information_score_turn = 0
                    required_patient_info = remove_punctuation(dialog['patient'])
                    infor_patient_info = remove_punctuation(data["Text"]["Patient Information"]) + ' '+ remove_punctuation(data["Text"]["Chief Complaint"]) + ' '+ remove_punctuation(data["Text"]["Present Illness"])  + ' '+ remove_punctuation(data["Text"]["Past Medical History"])
                    score = scorer.score(infor_patient_info, required_patient_info)
                    Information_score_turn += score['rouge1'].recall

                    # exam
                    exam_patient_info = remove_punctuation(data["Text"]["Examinations"])
                    score = scorer.score(exam_patient_info, required_patient_info)
                    Information_score_turn += score['rouge1'].recall

                    Information_score += Information_score_turn 

            for j in range(len(data['history'])):
                dialog = data['history'][j]
                Information_score_turn = 0
                if mm_token is True:
                    report_info = ''
                    mm_doctor_info =  remove_punctuation(dialog['doctor'])
                    
                    # information and information gain
                    infor_patient_info = remove_punctuation(data["Text"]["Patient Information"]) + ' '+ remove_punctuation(data["Text"]["Chief Complaint"]) + ' '+ remove_punctuation(data["Text"]["Present Illness"])  + ' '+ remove_punctuation(data["Text"]["Past Medical History"])
                    score = scorer.score(infor_patient_info, mm_doctor_info)
                    Information_score_turn += score['rouge1'].recall

                    # exam
                    exam_patient_info = remove_punctuation(data["Text"]["Examinations"])
                    score = scorer.score(exam_patient_info, mm_doctor_info)
                    Information_score_turn += score['rouge1'].recall
                    # 指标2
                    for image in image_name:
                        report_info += remove_punctuation(data['Image'][image])

                    score = scorer.score(report_info, remove_punctuation(dialog['doctor']))
                    Score2.append(score['rouge1'].recall)

                    # 指标3
                    s3_dialog.append(remove_punctuation(dialog['doctor']))
                    s3_dialog.append(remove_punctuation(nomm_datas[i]['history'][j]['doctor']))
                    Score3.append(distinct_n_corpus_level(s3_dialog, 1))

                    # 指标1
                    s1_dialog.append(remove_punctuation(dialog['doctor']))
                    s1_dialog.append(remove_punctuation(text_datas[i]['history'][j]['doctor']))
                    Score1.append(distinct_n_corpus_level(s1_dialog, 1))

                    break

                if dialog['state'] != 'D':
                    pattern = r'<\\image(.*?)>'
                    matches = re.findall(pattern,dialog['patient'], re.DOTALL)
                    if matches != []:
                        mm_token = True
                        image_name = matches
            # 指标4
            if mm_token is True:
                for image in data['Image']:
                    s4_report_info += remove_punctuation(data['Image'][image])
                score = scorer.score(s4_report_info, remove_punctuation(data["diagnosis_self"]))
                Score4.append(score['rouge1'].recall)

            if mm_token is not True:
                Score1.append(1)
                Score2.append(0)
                Score3.append(0)
                Score4.append(0)
        except:
            print('error')
    Score1 = round(np.average(Score1),4)*100
    Score2 = round(np.average(Score2),4)*100
    Score3 = round(np.average(Score3),4)*100
    Score4 = round(np.average(Score4),4)*100

    file_name = file_path.split("/")[-1].replace(".json", "")
    patient_model, doctor_model = file_name.split('_')
    print("%15s %15s %15s %15s %15s %15s" %(patient_model, doctor_model, Score1, Score2, Score3, Score4))  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/M3_30_Result_actor")
    parser.add_argument("--text-folder-path", type=str, default="/M3_30_Result_actor_text")
    parser.add_argument("--no-mm-folder-path", type=str, default="/M3_30_Result_actor_nomm")
    
    args = parser.parse_args()
    print("%15s %15s %15s %15s %15s %15s" %('Patient Model', 'Doctor Model', "Score1", "Score2", "Score3", "Score4")) 
    calculate_metric(args)