import os
import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from utils.general_utils import disable_torch_init

LOCAL_MODEL_PATHS = {
    'qwen-vl-chat':'/Qwen-VL-Chat',
    'Mini-InternVL-Chat-4B-V1-5':'/Mini-InternVL-Chat-4B-V1-5',
    'InternVL-Chat-V1-5':'/InternVL-Chat-V1-5',
    'glm-4-9b': '/glm-4-9b',
    'huatuo-vision-7b': '/HuatuoGPT-Vision-7B',
    'huatuo-vision-34b': '/HuatuoGPT-Vision-34B'
}

class Base_Model:
    def __init__(self):
        pass

    def postprocessed(self, outputs):
        return outputs.strip(" \n")

    def generate(self):
        pass

    def multiple_choice_selection(self):
        pass

    def log(self):
        pass

    def get_logit_bias(self, state_num=4):
        pass

class API_Model(Base_Model):
    def __init__(self, api_key, stop_ids):
        super().__init__()
        self.api_key = api_key
        self.stop_ids = stop_ids
    
    def generate(self):
        return super().generate()

class Local_Model(Base_Model):
    def __init__(self, model_name, stop_ids):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model.eval()

        stop_ids.append(torch.tensor([self.tokenizer.eos_token_id])) # </s>
        # pdb.set_trace()
        # stop_ids.append(torch.tensor([self.tokenizer("PATIENT")]))
        self.stop_ids = [s.cuda() for s in stop_ids]
        self.stop_criteria = KeywordsStoppingCriteria(self.stop_ids)
    
    def get_logit_bias(self, state_num=4):
        state_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        logit_bias = {}
        # pdb.set_trace()
        for i in range(state_num):
            logit_bias[self.tokenizer(state_list[i], add_special_tokens=False)["input_ids"][0]] = 100

        return logit_bias
    
    def generate(self, inputs):
        inputs = self.tokenizer([inputs])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=False,
                max_new_tokens=300,
                stopping_criteria=StoppingCriteriaList([self.stop_criteria]))
        
        for stop_wrod in self.stop_ids:
            if torch.all(stop_wrod == output_ids[0][-len(stop_wrod):]).item():
                output_ids = output_ids[:, :-len(stop_wrod)]
                break
     
        final_outputs = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.decode(final_outputs)

        return self.postprocessed(outputs)

    def multiple_choice_selection(self, inputs, logit_bias):
        logits_processor_list = LogitsProcessorList([
            LogitBiasLogitsProcessor(logit_bias),
        ])

        inputs = self.tokenizer([inputs])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=False,
                max_new_tokens=1,
                logits_processor=logits_processor_list,
            )
        
        final_outputs = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.decode(final_outputs)
        # pdb.set_trace()

        return outputs

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for keyword in self.keywords:
            if torch.all(keyword == input_ids[0][-len(keyword):]).item():
                return True
        return False

class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = logit_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        for index in self.logit_bias.keys():
            scores[:, index] += self.logit_bias[index]
        return scores


