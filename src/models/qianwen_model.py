import dashscope
from .base_model import API_Model, Local_Model, LOCAL_MODEL_PATHS, KeywordsStoppingCriteria, Base_Model
import pdb
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from utils.general_utils import disable_torch_init


class QianWen_Model(API_Model):
    def __init__(self, 
                 api_key="",
                 version="qwen_max",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        dashscope.api_key = api_key
        if version == "qwen_plus":
            self.version = dashscope.Generation.Models.qwen_plus
        elif version == "qwen_max":
            self.version = dashscope.Generation.Models.qwen_max
        elif version == "qwen_max_longcontext":
            self.version = dashscope.Generation.Models.qwen_max_longcontext
        elif version == "qwen-1.8b-chat":
            self.version = 'qwen-1.8b-chat'  
        else:
            print(f"version=={version} is not available in qianwen model!")
            raise NotImplementedError
    
    def generate(self, inputs):
        message = [{"role": "user", "content": inputs}]

        while True:
            try:
                response = dashscope.Generation.call(
                    self.version,
                    messages=message,
                    # set the random seed, optional, default to 1234 if not set
                    seed=0,
                    stop=self.stop_ids,
                    result_format='message',  # set the result to be "message" format.
                    temperature=0.01,
                )

                outputs = response["output"]["choices"][0]["message"]["content"]
                break
            except:
                print("Error, Retrying...")
                # import pdb
                # pdb.set_trace()

        return outputs

class QianWen_Vision_Model(API_Model):
    def __init__(self, 
                 api_key="",
                 version="qwen-vl-max",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        dashscope.api_key = api_key
        if version == "qwen-vl-max":
            self.version = 'qwen-vl-max'
        elif version == "qwen-vl-plus":
            self.version = 'qwen-vl-plus'
        elif version == "qwen_vl_chat_v1":
            self.version = dashscope.MultiModalConversation.Models.qwen_vl_chat_v1
        elif version == "qwen-vl-v1":
            self.version = dashscope.MultiModalConversation.Models.qwen_vl_v1
        else:
            print(f"version=={version} is not available in qianwen model!")
            raise NotImplementedError
    
    def generate(self, inputs, images=None):

        content = [{'text':inputs}]
        if images is not None:
            for image in images:
                content.append({'image':'file://' + image})
        message = [{"role": "user", "content": content}]


        while True:
            try:
                response = dashscope.MultiModalConversation.call(
                    model = self.version,
                    messages=message,
                    # set the random seed, optional, default to 1234 if not set
                    seed=0,
                    stop=self.stop_ids,
                    result_format='message',  # set the result to be "message" format.  
                    temperature=0.01,
                )
                if isinstance(self.version, str) and self.version != 'qwen-vl-chat-v1':
                    outputs = response["output"]["choices"][0]["message"]["content"][0]['text']
                else:
                    outputs = response["output"]["choices"][0]["message"]["content"]

                break
            except:
                print("Error, Retrying...")
                import pdb
                

        return outputs

    
    # dashscope.MultiModalConversation.call( model = self.version,messages=message,seed=0,stop=self.stop_ids,result_format='message',  temperature=0.01, )

class QianWen_Vision_Local_Model(Base_Model):
    def __init__(self, 
                 model_name="qwen-vl-chat",
                 stop_ids=[]):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, fp16 = True, device_map="cuda", trust_remote_code=True)
        self.model.eval()

        stop_ids.append(torch.tensor([151643])) # </s>
        # pdb.set_trace()
        # stop_ids.append(torch.tensor([self.tokenizer("PATIENT")]))
        self.stop_ids = [s.cuda() for s in stop_ids]
        self.stop_criteria = KeywordsStoppingCriteria(self.stop_ids)
    
    def generate(self, inputs, images=None):

        content = [{'text':inputs}]
        if images is not None:
            for image in images:
                content.append({'image': image})

        message = self.tokenizer.from_list_format(content)

        while True:
            try:
                response, history = self.model.chat(self.tokenizer, query=message, history=None)

                outputs = response
                break
            except:
                print("Error, Retrying...")
                import pdb
                

        return outputs