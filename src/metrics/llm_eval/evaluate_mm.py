import openai
from openai import OpenAI
import json
from tools.transform_caseinfo import *
from jsonpath import jsonpath
# import os
import time
# import shutil
from tools.path import *

import tools.evaluateModel as em



class Evaluate_mm(em.OpenaiEvaluate):

    def evaluate_base(self, addr_model, addr_result, ne_addr_model, nm_addr_model, cnt_eval, time_str = time.strftime("-%Y-%m-%d-%H",time.localtime())):
        case_normal = json.load(open(ne_addr_model, 'r', encoding='utf-8'))
        case_text_only = json.load(open(addr_model, 'r', encoding='utf-8'))
        case_nomm = json.load(open(nm_addr_model, 'r', encoding='utf-8'))

        format_path_result = addr_result[:-4]+time_str+'.txt'
        count = 1
        for case_data, case_data_text, case_data_nomm in zip(case_normal , case_text_only, case_nomm):
            if count==cnt_eval:
                break
            with open(format_path_result, 'a', encoding='utf-8') as efile:
                efile.write('\n--------------------------'+'\ncase:'+ str(count)+"\n")
            count+=1
            status=0
            for prompt in self.evaluate_prompt:
                if status == 0:
                    mm_data, text_data = get_mm(case_data, case_data_text)
                    if(mm_data == "invalid"):
                        with open(format_path_result, 'a', encoding='utf-8') as efile:
                            efile.write("\n\nNo valid pic input\nscore: 1\n\nscore: 1")
                    else:
                        format_data = "pic data:\n"+mm_data+"\ntext data:\n"+text_data
                        self.evaluate(prompt, format_data, format_path_result)
                elif status == 1:
                    mm_data, nomm_data = get_mm(case_data, case_data_nomm)
                    if(mm_data == "invalid"):
                        with open(format_path_result, 'a', encoding='utf-8') as efile:
                            efile.write("\n\nNo valid pic input\nscore: 0")
                    else:
                        format_data = "pic data:\n"+mm_data+"\nnopic data:\n"+nomm_data
                        self.evaluate(prompt, format_data, format_path_result)
                status+=1

    def doEva_muti(self, text_model, text_result, normal_model, nomm_model, cnt_eval = 30):
        cnt_eval+=1
        for addr_model, addr_result, ne_addr_model, nm_addr_model in zip(text_model,text_result, normal_model, nomm_model):
            self.evaluate_base(addr_model, addr_result, ne_addr_model, nm_addr_model, cnt_eval)
            time.sleep(5)

    def doEva_single(self, text_model, text_result, normal_model, nomm_model, cnt_eval = 30):
        cnt_eval+=1
        self.evaluate_base(text_model, text_result, normal_model, nomm_model, cnt_eval)
            

    

if __name__ == '__main__':
    # openaimodel = openaiModel()
    prompt = './data/mm_prompt.json'
    eva = Evaluate_mm(prompt)
    eva.doEva_muti(TE.path_model, TE.path_result, NE.path_model, NME.path_model)
    