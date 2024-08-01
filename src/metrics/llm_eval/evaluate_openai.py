import openai
import json
from tools.transform_caseinfo import *
from jsonpath import jsonpath
import tools.wash_nopic as wash_nopic
# import os
import tools.evaluateModel as em
import time
# import shutil
from tools.path import *

class Evaluate_normal(em.OpenaiEvaluate):
    
    def evaluate_base(self, addr_model, addr_result, cnt_eval, time_str = time.strftime("-%Y-%m-%d-%H",time.localtime())):
        case_all = json.load(open(addr_model, 'r', encoding='utf-8'))
        format_path_result = addr_result[:-4]+time_str+'.txt'
        count = 1
        for case_data in case_all:
            if count==cnt_eval:
                break
            case_format = transcase(case_data)
            with open(format_path_result, 'a', encoding='utf-8') as efile:
                efile.write('\n--------------------------'+'\ncase:'+ str(count)+"\n")
            count+=1
            for prompt in self.evaluate_prompt:
                self.evaluate(prompt, case_format, format_path_result)

    def doEva_muti(self, addr_model, addr_result, cnt_eval = 30):
        cnt_eval+=1
        # time_str = time.strftime("-%Y-%m-%d-%H",time.localtime()) 
        for addr_model, addr_result in zip(addr_model,addr_result):
            self.evaluate_base(addr_model, addr_result, cnt_eval)
            time.sleep(5)

    def doEva_single(self, addr_model, addr_result, cnt_eval = 30):
        cnt_eval+=1
        self.evaluate_base(addr_model, addr_result, cnt_eval)

                                           
if __name__ == '__main__':
    # openaimodel = openaiModel()
    prompt = './data/prompt_en.json'
    eva = Evaluate_normal(prompt)
    eva.doEva_muti(NE.path_model, NE.path_result)
