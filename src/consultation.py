import json
import os
import openai
import pdb
import re
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
from random import sample
import argparse
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
from utils.general_utils import mkdir
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.agents import Doctor_Agent, Patient_Agent, StateDetect_Agent, Dignosis_Agent
from models import get_model, API_Model, QianWen_Model


def agent_initialization(args):
    # patient agent
    patient_conv = get_patient_template(args.mode, args.patient_model).copy()
    patient_model = get_model(args.patient_model, patient_conv.stop_ids)
    patient_agent = Patient_Agent(args, patient_model, patient_conv)
    print(f"[Patient Conv] {type(patient_conv)}")
    print(f"[Patient Model] {type(patient_model)}")
    print(f"[Patient Agent] {type(patient_agent)}")

    # doctor agent
    doctor_conv = get_doctor_template(args.mode, args.doctor_model).copy()
    doctor_model = get_model(args.doctor_model, doctor_conv.stop_ids)
    doctor_agent = Doctor_Agent(args, doctor_model, doctor_conv)
    print(f"[Doctor Conv] {type(doctor_conv)}")
    print(f"[Doctor Model] {type(doctor_model)}")
    print(f"[Doctor Agent] {type(doctor_agent)}")

    # state agent
    if args.state_model == args.patient_model or args.state_model is None:
        state_model = patient_model
        state_agent = StateDetect_Agent(args, patient_model, conv=None, state_num=5)
    else:
        state_model = get_model(args.state_model, stop_ids=[])
        state_agent = StateDetect_Agent(args, state_model, conv=None, state_num=5)
    print(f"[State Model] {type(state_model)}")
    print(f"[State Agent] {type(state_agent)}")
    
    # diagnosis agent
    if args.diagnosis_model == args.doctor_model or args.diagnosis_model is None:
        diagnosis_model = doctor_model
        diagnosis_agent = Dignosis_Agent(args, doctor_model, conv=None, candidates_num=5)
    else:
        diagnosis_model = get_model(args.diagnosis_model, stop_ids=[])
        diagnosis_agent = Dignosis_Agent(args, diagnosis_model, conv=None, candidates_num=5)
    print(f"[Diagnosis Model] {type(diagnosis_model)}")
    print(f"[Diagnosis Agent] {type(diagnosis_agent)}")

    return patient_agent, doctor_agent, state_agent, diagnosis_agent

def generate_forward(agent, datas, *kwargs):
    if isinstance(agent.model, API_Model):
        if isinstance(agent.model, XingHuo_Model) or isinstance(agent.model, QianWen_Model) or isinstance(agent.model, YiYan_Model):
            pool = ThreadPool(processes=1)
        else:
            pool = ThreadPool(processes=args.workers)
            
        pool.starmap(agent.generate, [[data] + list(kwargs) for data in datas])
        pool.close()
        pool.join()
    else:
        for data in datas:
            agent.generate(data, *kwargs)
    
    agent.log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-name", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)

    parser.add_argument("--patient-prompt-id", type=str, default="base_v1_ch")
    parser.add_argument("--patient-model", type=str, default="qianwen")
    parser.add_argument("--patient-history-len", type=int, default=1)

    parser.add_argument("--doctor-prompt-id", type=str, default="base_v1_zh")
    parser.add_argument("--doctor-model", type=str, default="gpt4")

    parser.add_argument("--state-model", type=str, default='base_v1_zh')
    parser.add_argument("--exam-model", type=str, default='base_v1_zh')
    parser.add_argument("--diagnosis-model", type=str, default=None)

    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--actor-simulator", action="store_true", default=False)
    parser.add_argument("--only-text", action="store_true", default=False)
    parser.add_argument("--no-mm", action="store_true", default=False)
    parser.add_argument("--only-diagnosis", action="store_true", default=False)
    parser.add_argument("--actor-plus", type=str, default='None')
    parser.add_argument("--cot", action="store_true", default=False)
    parser.add_argument("--cot-file-path", type=str, default='None')
    parser.add_argument("--zero-cot", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[Input File] {args.input_file_name}")
    print(f"[Output File] {args.output_file_name}")
    print(f"[Cot File] {args.cot_file_path}")
    args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
    args.output_file_name = args.output_file_name.rsplit(".", 1)[0]
    args.cot_file_path = args.cot_file_path.rsplit(".", 1)[0]

    # mkdir
    mkdir(args.output_file_name)

    # data prepare process
    datas, _ = data_initialization(args)
    if datas == []:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            for q in origin_questions:
                if "history" not in q.keys():
                    q["history"] = []
                datas.append(q)
    
    with open(f"{args.cot_file_path}.json", "r", encoding="utf-8") as f:
        cot_data = json.load(f)

    # agent
    patient_agent, doctor_agent, state_agent, diagnosis_agent = agent_initialization(args)

    if args.debug:
        datas = datas[:1]
        args.workers = 1
        # datas[0]["history"] = []

    # datas = datas[:5]
    total_count = len(datas)
    turn_pos = max([len(datas[i]['history']) for i in range(len(datas))])
    # pdb.set_trace()
    for i in range(args.max_turn):
        if i < turn_pos and args.only_text is False and args.no_mm is False:
            continue
        ## Doctor genertaion
        bar = Bar(f'Processing Turn [{i}] Doctor', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(doctor_agent, datas, cot_data, i, bar)
        bar.finish()

        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

        ## State Recognition
        bar = Bar(f'Processing Turn [{i}] State Detection I', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(state_agent, datas, i, "stageI", bar)
        bar.finish()

        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

        bar = Bar(f'Processing Turn [{i}] State Detection II', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(state_agent, datas, i, "stageII", bar)
        bar.finish()

        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

        bar = Bar(f'Processing Turn [{i}] State Detection III', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(state_agent, datas, i, "stageIII", bar)
        bar.finish()

        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

        ## Patient generation
        bar = Bar(f'Processing Turn [{i}] Patient', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(patient_agent, datas, i, bar)
        bar.finish()


        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

    # Diagnosis
    bar = Bar(f'Processing Diagnosis', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    generate_forward(diagnosis_agent, datas, bar)
    bar.finish()

    if not args.debug:
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)