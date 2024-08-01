import json
import openai
from openai import OpenAI
import time
import pytz
from datetime import datetime

class OpenaiEvaluate:
    openai.api_key = ""
    prompt_path = ""
    evaluate_prompt = ""
    log_file = "./logs/openai.json"
    cost_log = {
        "message ": "",
        "input_tokens" : 0,
        "output_tokens": 0 ,
        "dollar_cost": 0,
        "time_end": ""
    }

    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        self.evaluate_prompt = json.load(open(prompt_path, 'r', encoding='utf-8'))

    def log(self, message=None):
        self.cost_log["message"] = message
        with open(self.log_file, "r")as files:
            json_data = json.load(files)
            json_data.append(self.cost_log)

        with open(self.log_file, "w") as f:
            try:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            except:
                print("Error in log")
            
    def get_time(self):
        # 设置时区为中国时间
        china_timezone = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(china_timezone)
        return f"{current_time.year}-{current_time.month}-{current_time.day} {current_time.hour}:{current_time.hour}:{current_time.minute}:{current_time.second}"

    def update_log(self, message):
        self.cost_log["input_tokens"] = message.usage.prompt_tokens
        self.cost_log["output_tokens"] = message.usage.completion_tokens
        self.cost_log["dollar_cost"] = self.cost_log["input_tokens"] * 5e-6 + self.cost_log["output_tokens"] * 1.5e-5
        self.cost_log["time_end"] = self.get_time()

    def evaluate(self, evaluate_prompt, case_data, path_result):
        client = OpenAI(
            api_key=openai.api_key
        )

        messages = [{'role': 'system', 'content': evaluate_prompt["prompt"]},
                {'role': 'user', 'content': case_data},]
        t_start = time.perf_counter()
        while True:
            try:
                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=messages,
                                    )
                outputs = response.choices[0].message.content
                self.update_log(response)
                if outputs:
                    break 
                else:
                    print("Output is none, Retrying...")
            except openai.RateLimitError as e:
                print(e)
                t_rest = 60 - ( (time.perf_counter() - t_start) % 60 )
                print(f"surpass the tpm limits, wait for {t_rest} seconds...")
                time.sleep(t_rest)
                t_start = time.perf_counter()
            except openai.APITimeoutError as e:
                print("Timeout Error, Retrying...")
            except openai.APIConnectionError as e:
                print("Connect Error, Retrying...")
 
        with open(path_result, 'a', encoding='utf-8') as efile:
            efile.write('\n\n'+ outputs) 

        self.log(outputs) 