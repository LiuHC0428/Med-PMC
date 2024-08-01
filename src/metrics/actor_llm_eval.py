import time
import os
import pdb
import json
import openai
from datetime import datetime
from openai import OpenAI
import pytz
import base64
import torch
import random
import numpy as np


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 


class OpenAI_VISION_Model():
    def __init__(self, 
                 model_type="gpt-4o",
                api_key="",
                 stop_ids=[]):
        super().__init__()
        self.t_start = time.perf_counter()
        self.model_type = model_type
        self.client = OpenAI(
            api_key=api_key
        )
        self.cost_log = {}
        self.cost_log["input_tokens"] = 0
        self.cost_log["output_tokens"] = 0

        self.log_file = '/openai.json'
        self.api_key = api_key
        self.stop_ids = stop_ids
    
    def log(self, message=None):
        self.cost_log["message"] = message
        with open(self.log_file, "w") as f:
            json.dump(self.cost_log, f, indent=4, ensure_ascii=False)

    def get_time(self):
        # 设置时区为中国时间
        china_timezone = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(china_timezone)
        return f"{current_time.year}-{current_time.month}-{current_time.day} {current_time.hour}:{current_time.hour}:{current_time.minute}:{current_time.second}"

    def update_log(self, message):
        self.cost_log["input_tokens"] += message.usage.prompt_tokens
        self.cost_log["output_tokens"] += message.usage.completion_tokens
        if self.model_type == "gpt-4o-2024-05-13":
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 5e-6 + self.cost_log["output_tokens"] * 1.5e-5
        elif self.model_type == "gpt-4-turbo-2024-04-09": 
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 1e-5 + self.cost_log["output_tokens"] * 3e-5
        self.cost_log["time_end"] = self.get_time()
    
    def generate(self, inputs, images=None , max_tokens=1000):

        content = [{"type": "text",'text':inputs}]
        if images is not None:
            for image in images:
                with open(image,'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
        message = [{"role": "user", "content": content}]

        self.client = OpenAI(
            api_key=self.api_key
        )
        
        while True:
            try:
                # pdb.set_trace()
                # client = OpenAI(api_key=self.api_key)
                completion = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=message,
                    temperature=0,
                    seed=0,
                    max_tokens=max_tokens,
                    stop=self.stop_ids,
                )
                outputs = completion.choices[0].message.content

                #  outputs = self.client.chat.completions.create(model=self.model_type,messages=message,temperature=0,seed=0,max_tokens=max_tokens,stop=self.stop_ids,).choices[0].message.content
                
                self.update_log(completion)
                if outputs:
                    break 
                else:
                    print("Output is none, Retrying...")
            except openai.RateLimitError as e:
                print(e)
                t_rest = 60 - ( (time.perf_counter() - self.t_start) % 60 )
                print(f"surpass the tpm limits, wait for {t_rest} seconds...")
                time.sleep(t_rest)
                self.t_start = time.perf_counter()
            except openai.APITimeoutError as e:
                print("Timeout Error, Retrying...")
            except openai.APIConnectionError as e:
                print("Connect Error, Retrying...")
         
        return outputs

def get_actor_prompt(prompt_id):
    prompt_id = prompt_id.lower()

    if 'farmer' in prompt_id:
        prompt_id = 'Farmer'
    elif 'student'  in prompt_id:
        prompt_id = 'Student'
    elif 'worker'  in prompt_id and 'office' not in prompt_id:
        prompt_id = 'Worker'
    elif 'doctor'  in prompt_id:
        prompt_id = 'Doctor'
    else:
        prompt_id = 'Office-worker'

    return prompt_id

eval_model = OpenAI_VISION_Model()


set_random_seed(23)
input_prompts = []
output_scores = []
folder_path = '/M3_30_Result_actor'
prompt = "<Requirement>: The existing output is converted into an output with character characteristics according to different occupations, and the quality of the conversion statement is evaluated according to the semantic consistency and character feature richness. \
        The total score is 10 points, every aspect is 5 points, please output the score directly and do not output the analyze. \
        The output format is: <Semantic Consistency Score>: n <Character Feature Score>: n \n"

for path, dir_lst, file_lst in os.walk(folder_path):
    for file_name in file_lst:
        file_path = os.path.join(folder_path,file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            datas = json.load(f)

        tensor_chosen = torch.randint(0, 10, (1, 3))

        for number in tensor_chosen[0]:
            turn_chosen = torch.randint(0, len(datas[number]['history'])-1, (1, 1))[0]
            for i in range(5):
                if 'standard_patient' in datas[number]['history'][turn_chosen].keys():
                    standard_patient = datas[number]['history'][turn_chosen]['standard_patient']
                    actor = datas[number]['history'][turn_chosen]['patient']
                    prompt_id = get_actor_prompt(datas[number]['Text']['Patient Information'])
                    input_prompts.append('<career>:' + prompt_id + '<original output>:' + standard_patient + '<transfered output>:' + actor)
                    break
                else:
                    turn_chosen = torch.randint(0, len(datas[number]['history'])-1, (1, 1))[0]


for input_prompt in input_prompts:
    message = prompt + input_prompt
    score = eval_model.generate(message)
    output_scores.append(score)

with open('/M3/Eval_result/actor_llm_eval.json', "w", encoding="utf-8") as f:
    json.dump(output_scores, f, indent=4, ensure_ascii=False)

