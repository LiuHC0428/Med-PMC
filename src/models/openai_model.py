import time
import os
import pdb
import json
import openai
from datetime import datetime
from openai import OpenAI
from .base_model import API_Model
import pytz
import base64

class OpenAI_Model(API_Model):
    def __init__(self, 
                 model_type="gpt-3.5-turbo-1106",
                api_key="",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        self.t_start = time.perf_counter()
        self.model_type = model_type
        self.client = OpenAI(
            api_key=api_key
        )
    
    def get_logit_bias(self, state_num=4):
        return {(32+i):100 for i in range(state_num)}
    
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
        if self.model_type == "gpt-3.5-turbo-1106":
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 1e-6 + self.cost_log["output_tokens"] * 2e-6
        elif self.model_type == "gpt-4-1106-preview": 
            self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 1e-5 + self.cost_log["output_tokens"] * 3e-5
        self.cost_log["time_end"] = self.get_time()
    
    def generate(self, inputs, max_tokens=300):
        message = [{"role": "user", "content": inputs}]

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
    
    def multiple_choice_selection(self, inputs, logit_bias):
        message = [{"role": "user", "content": inputs}]

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
                    logit_bias=logit_bias,
                    temperature=0.0,
                    seed=0,
                    max_tokens=1,
                    )   
                self.update_log(completion)
                break 
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
            
         
        outputs = completion.choices[0].message.content
        return outputs

class OpenAI_VISION_Model(API_Model):
    def __init__(self, 
                 model_type="gpt-4o",
                api_key="",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        self.t_start = time.perf_counter()
        self.model_type = model_type
        self.client = OpenAI(
            api_key=api_key
        )
        self.cost_log = {}
        self.cost_log["input_tokens"] = 0
        self.cost_log["output_tokens"] = 0

        self.log_file = '/openai.json'
    
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
                # if outputs:
                #     break 
                # else:
                #     pdb.set_trace()
                #     print("Output is none, Retrying...")
                break
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