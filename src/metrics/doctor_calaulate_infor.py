import os
import re
import pdb
import json
import argparse
import numpy as np
import spacy
import xlwt
from rouge_score import rouge_scorer
from distinct_utils import distinct_n_corpus_level
# from ..utils.openai_utils import split_chinese_medicalinfo_and_question

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


    print("%15s %15s %15s %15s %15s %15s %15s" %('Patient Model', 'Doctor Model',  "Coverage_Rouge1", "Medical_Rate", "Average_Turn", "Average_Length", "Information_gain")) 
    for file in files:
        if file.find('_') != -1:
        # if file.endswith(r"\d+_multiple_6\.json"):
            if re.search(pattern, file):
                eval_dialogue(os.path.join(args.folder_path, file))

def eval_dialogue(file_path):
    with open(file_path, "r") as f:
        datas = json.load(f)
    
    Coverage_Rouge1 = []
    Average_Turn = []
    Average_Length = []
    Medical_Rate = []
    Information_gain = []

    for data in datas:
        Coverage_Rouge1_turn = []
        Information_gain_turn = []
        Average_Length_turn = []
        non_medical_rate = 0
        Information_score = 0
        total_required_patient_info = ''
        total_required_info = ''
        total_infor_patient_info = remove_punctuation(data["Text"]["Patient Information"]) + ' '+ remove_punctuation(data["Text"]["Chief Complaint"]) + ' '+ remove_punctuation(data["Text"]["Present Illness"])  + ' '+ remove_punctuation(data["Text"]["Past Medical History"]) 
        total_info = remove_punctuation(data["Text"]["Patient Information"]) + ' '+ remove_punctuation(data["Text"]["Chief Complaint"]) + ' '+ remove_punctuation(data["Text"]["Present Illness"])  + ' '+ remove_punctuation(data["Text"]["Past Medical History"])  + ' '+ remove_punctuation(data["Text"]['Examinations']) 
        
        for dialog in data['history']:
            if dialog['state'] != 'D':
                # information and information gain
                if 'standard_patient' in dialog.keys():
                    total_required_patient_info += remove_punctuation(dialog['standard_patient'])
                    total_required_info += remove_punctuation(dialog['standard_patient'])
                else:
                    total_required_info += remove_punctuation(dialog['patient'])
            

                # 是否全程医疗问题
                if dialog['state'].find('C') != -1:
                    non_medical_rate += 1

                Average_Length_turn.append(len(dialog["doctor"].split(" ")))

        score = scorer.score(total_infor_patient_info, total_required_patient_info)
        Coverage_Rouge1_turn.append(score['rouge1'].recall)

        score = scorer.score(total_info, total_required_info)
        total_infor_gain = score['rouge1'].recall

        required_info = ''
        for dialog in data['history']:
            if dialog['state'] != 'D':
                if 'standard_patient' in dialog.keys():
                    required_info += remove_punctuation(dialog['standard_patient'])
                else:
                    required_info += remove_punctuation(dialog['patient'])

                score = scorer.score(total_info, required_info)
                infor_gain = score['rouge1'].recall

                infor_gain_split = infor_gain

                Information_gain_turn.append(infor_gain_split)
                
        
        Coverage_Rouge1.append(np.average(Coverage_Rouge1_turn))
        Medical_Rate.append(1-non_medical_rate/len(data["history"]))
        Average_Turn.append(len(data["history"]))
        Average_Length.append(np.average(Average_Length_turn))
        if len(data['history']) == 3:
            Information_gain.append(Information_gain_turn)
        elif len(data['history']) == 2:
            a = 0    
        else:
            Information_gain.append(Information_gain_turn)
    
    Information_final_turn = []
    for i in range(10):
        Information_gain_split = []
        for j in range(len(datas)):
            try:
                Information_gain_split.append(Information_gain[j][i])
            except:
                continue
        Information_final_turn.append(np.average(Information_gain_split))

    file_name = file_path.split("/")[-1].replace(".json", "")
    doctor_model, patient_model = file_name.split('_')
    
    f = xlwt.Workbook('encoding = utf-8') #设置工作簿编码
    sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表
    list1 = Information_final_turn
    for i in range(len(list1)):
        sheet1.write(0,i,list1[i]) #写入数据参数对应 行, 列, 值
    f.save('/M3/Eval_result/infor_gain_'+doctor_model+'.xls')#保存.xls到当前工作目录

                                                                            
    # print("%15s %15s %15s %15s %15s %15s %15s" %(patient_model, doctor_model, Coverage_Rouge1, Medical_Rate, Average_Turn, Average_Length, Information_gain))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/M3_30_Result_standard")
    
    args = parser.parse_args()
    calculate_metric(args)