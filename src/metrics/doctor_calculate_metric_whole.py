import os
import re
import pdb
import json
import argparse
import numpy as np
import spacy
from rouge_score import rouge_scorer
from distinct_utils import distinct_n_corpus_level
# from ..utils.openai_utils import split_chinese_medicalinfo_and_question

nlp = spacy.load("en_core_web_sm")
scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

def remove_punctuation(text):
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation

def calculate_metric(args):
    files = os.listdir(args.folder_path)
    files.sort() 
    pattern = r'\.json' 

    print("%15s %15s %15s %15s %15s %15s" %('Patient Model', 'Doctor Model', "Dig_Accuracy", "Coverage_Rouge1", "Knowledge_Recall", "tre_Accuracy")) 
    for file in files:
        if file.find('_') != -1:
        # if file.endswith(r"\d+_multiple_6\.json"):
            if re.search(pattern, file):
                eval_report(os.path.join(args.folder_path, file))

    print("%15s %15s %15s %15s %15s %15s %15s" %('Patient Model', 'Doctor Model',  "Coverage_Rouge1", "Medical_Rate", "Average_Turn", "Average_Length", "Information_gain")) 
    for file in files:
        if file.find('_') != -1:
        # if file.endswith(r"\d+_multiple_6\.json"):
            if re.search(pattern, file):
                eval_dialogue(os.path.join(args.folder_path, file))

def eval_report(file_path):
    with open(file_path, "r") as f:
        datas = json.load(f)
    
    Dig_Accuracy = []
    Coverage_Rouge1 = []
    Knowledge_Recall = []
    Tre_Accuracy = []
    for data in datas:

        Dig_Accuracy.append(scorer.score(data["Text"]["Diagnosis"], data["diagnosis_self"])['rougeL'].recall)
        
        # Chief Complaint + Present Illness + Past Medical History
        required_patient_info = data["diagnosis_self"]
        patient_info = remove_punctuation(data["Text"]["Chief Complaint"]) + ' '+ remove_punctuation(data["Text"]["Present Illness"])  + ' '+ remove_punctuation(data["Text"]["Past Medical History"]) 
        score = scorer.score(patient_info, required_patient_info)
        Coverage_Rouge1.append(score['rouge1'].recall)

        # Examinations
        required_patient_info = data["diagnosis_self"]
        patient_info = remove_punctuation(data["Text"]["Examinations"])
        parse_info = nlp(patient_info)
        reference_entity_list = [ent.text for ent in parse_info.ents]
        if len(reference_entity_list) > 0:
            k_match = 0
            for ent in reference_entity_list:
                if ent in required_patient_info:
                    k_match += 1
            Knowledge_Recall.append(k_match / len(reference_entity_list))

        # Treatment
        Tre_Accuracy.append(scorer.score(data["Text"]["Treatment"], data["diagnosis_self"])['rouge1'].recall)
        # if data["Text"]["Treatment"] in data["diagnosis_self"]:
        #     Tre_Accuracy.append(1)
        # else:
        #     Tre_Accuracy.append(0)

    
    Dig_Accuracy = round(np.average(Dig_Accuracy),4)*100
    Coverage_Rouge1 = round(np.average(Coverage_Rouge1),4)*100
    Knowledge_Recall = round(np.average(Knowledge_Recall),4)*100
    Tre_Accuracy = round(np.average(Tre_Accuracy),4)*100

    file_name = file_path.split("/")[-1].replace(".json", "")
    patient_model, doctor_model = file_name.split('_')
    print("%15s %15s %15s %15s %15s %15s" %(patient_model, doctor_model, Dig_Accuracy, Coverage_Rouge1, Knowledge_Recall, Tre_Accuracy))  


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

                infor_gain_split = total_infor_gain - infor_gain

                Information_gain_turn.append(infor_gain_split)
                
        
        Coverage_Rouge1.append(np.average(Coverage_Rouge1_turn))
        Medical_Rate.append(1-non_medical_rate/len(data["history"]))
        Average_Turn.append(len(data["history"]))
        Average_Length.append(np.average(Average_Length_turn))
        Information_gain.append(np.average(([Information_gain_turn[i]-Information_gain_turn[i+1] for i in range(len(Information_gain_turn)-1)][1:])))


    Coverage_Rouge1 = round(np.average(Coverage_Rouge1),4)*100
    Medical_Rate = round(np.average(Medical_Rate),4)
    Average_Turn = round(np.average(Average_Turn),4)
    Average_Length = round(np.average(Average_Length),4)
    Information_gain = round(np.average(Information_gain),4)*100
                                                                            
    file_name = file_path.split("/")[-1].replace(".json", "")
    patient_model, doctor_model = file_name.split('_')
    print("%15s %15s %15s %15s %15s %15s %15s" %(patient_model, doctor_model, Coverage_Rouge1, Medical_Rate, Average_Turn, Average_Length, Information_gain))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/M3_30_Result_standand_cot_actor")
    
    args = parser.parse_args()
    calculate_metric(args)