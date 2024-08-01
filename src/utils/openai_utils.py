import os
import json
import numpy as np

def data_initialization(args):
    # if os.path.exists(f"{args.output_file_name}.npy") and os.path.exists(f"{args.output_file_name}_temp.npy"):
    #     seed_tasks = np.load(f"{args.output_file_name}.npy", allow_pickle=True).tolist()
    #     seed_tasks_temp = np.load(f"{args.output_file_name}_temp.npy", allow_pickle=True).tolist()
    #     if len(seed_tasks) > len(seed_tasks_temp):
    #         seed_idx = [s["id"] for s in seed_tasks]
    #     else:
    #         seed_idx = [s["id"] for s in seed_tasks_temp]
    # elif os.path.exists(f"{args.output_file_name}.npy"):
    #     seed_tasks = np.load(f"{args.output_file_name}.npy", allow_pickle=True).tolist()
    #     seed_idx = [s["id"] for s in seed_tasks]
    # elif os.path.exists(f"{args.output_file_name}_temp.npy"):
    #     seed_tasks = np.load(f"{args.output_file_name}_temp.npy", allow_pickle=True).tolist()
    #     seed_idx = [s["id"] for s in seed_tasks]
    # elif os.path.exists(f"{args.output_file_name}.json"):
    #     with open(f"{args.output_file_name}.json", "r", encoding="utf-8") as f:
    #         seed_tasks = json.load(f)
    #     seed_idx = [s["id"] for s in seed_tasks]
    # else:
    #     seed_tasks = []
    #     seed_idx = []

    if os.path.exists(f"{args.output_file_name}.json"):
        with open(f"{args.output_file_name}.json", "r", encoding="utf-8") as f:
            seed_tasks = json.load(f)
        seed_idx = [s["id"] for s in seed_tasks]
    else:
        seed_tasks = []
        seed_idx = []
    
    return seed_tasks, seed_idx



def split_chinese_medicalinfo_and_question(total_question):
    total_question = total_question.strip()
    if total_question[-1] == "。":
        total_question = total_question[:-1]

    if total_question.rfind("。") > total_question.rfind("，"):
        medicalinfo, question = total_question.rsplit("。", 1)
    else:
        medicalinfo, question = total_question.rsplit("，", 1)
    
    return medicalinfo, question