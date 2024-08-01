import dataclasses
import torch
from enum import auto, Enum
from utils.general_utils import get_value
from typing import List, Tuple, Any
import pdb


class SeparatorStyle(Enum):
    """Different separator style."""
    VICUNA = auto()
    LLAMA2 = auto()
    FALCON = auto()
    CHATINTERN = auto()
    CHATGLM3 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    stop_ids: List[torch.tensor]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.VICUNA
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None
    
    def system_prompt_init(self, prompt):
        self.system = self.system.format(prompt=prompt)

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.VICUNA:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.FALCON:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "

            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += role + ": " + message + " "
                    else:
                        ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ": " + message + seps[i % 2] + "\n"
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def init_history(self, history_list, turn=-1, latest=True, first_key="question", second_key="answer"):
        if turn < 0 or turn > len(history_list):
            pass
        elif turn == 0:
            history_list = []
        else:
            if latest:
                history_list = history_list[-turn:]
            else:
                history_list = history_list[:turn]

        for history in history_list:
            if first_key in history.keys():
                self.messages.append([self.roles[0], history[first_key]])
                if second_key in history.keys():
                    self.messages.append([self.roles[1], history[second_key]])
    
    def pop_message(self):
        self.messages = self.messages[:-1]
    
    def clean_message(self):
        self.messages = []

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            stop_ids=self.stop_ids,
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

conv_vicuna_v1_1 = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([3970, 1783, 1955]), torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 29925, 1299, 29902, 3919]), torch.tensor([349, 1299, 29902, 3919]), torch.tensor([13, 13, 29925, 1299, 29902, 3919]),]
)

conv_chatgpt = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["DOCTOR", "PATIENT"]
)

conv_chatgpt_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["[医生]", "[患者]"]
)

conv_bloomz_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[torch.tensor([62, 25967, 64]), torch.tensor([62, 26122, 64])]
)

conv_gpt4 = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["DOCTOR", "PATIENT"]
)

conv_gpt4_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[]
)

conv_falcon = Conversation(
        system="System: {prompt}",
        roles=("DOCTOR", "PATIENT"),
        messages=[],
        sep_style=SeparatorStyle.FALCON,
        offset=0,
        sep="\n",
        sep2="<|endoftext|>",
        stop_ids=[torch.tensor([4310, 4274, 1951]), torch.tensor([8769, 4274, 1951]), torch.tensor([40392]), torch.tensor([19363]), torch.tensor([38293]), torch.tensor([12775])]
    )

conv_llama2 = Conversation(
    system="[INST] <<SYS>>\n{prompt}<</SYS>>\n\n",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    sep_style=SeparatorStyle.LLAMA2,
    offset=0,
    sep=" ",
    sep2=" </s><s>",
    stop_ids=[torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 3970, 1783, 1955]), torch.tensor([29871, 11662, 1783, 1955]), torch.tensor([13,  3970,  1783,  1955, 29901]), torch.tensor([ 13, 29925, 1299, 29902, 3919]), torch.tensor([1299, 29902, 3919])]
)

conv_llama2_zh = Conversation(
    system="[INST] <<SYS>>\n{prompt}<</SYS>>\n\n",
    roles=("[医生]", "[患者]"),
    messages=[],
    sep_style=SeparatorStyle.LLAMA2,
    offset=0,
    sep=" ",
    sep2=" </s><s>",
    stop_ids=[torch.tensor([232, 143, 190, 30486])]
)

conv_baichuan = Conversation(
    system="{prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([5946])]
)

conv_baichuan_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([1633, 5946, 31295]), torch.tensor([1633, 4304, 31295])]
)

conv_internlm = Conversation(
    system="{prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([68305])]
)

conv_internlm_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([336, 68305, 332]), torch.tensor([336, 68049, 332])]
)

conv_chatglm3 = Conversation(
    system="<|system|>\n {prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATGLM3,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([[790, 32718, 30996]]), torch.tensor([[790, 32016, 30996]])]
)
conv_chatglm3_zh = Conversation(
    system="<|system|>\n {prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATGLM3,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([[790, 32718, 30996]]), torch.tensor([[790, 32016, 30996]])]
)

conv_templates = {
    "vicuna": conv_vicuna_v1_1,
    "falcon": conv_falcon,
    "llama": conv_llama2,
    "chatgpt": conv_chatgpt,
    "gpt4": conv_chatgpt,
    "chatglm3": conv_chatglm3,
    "baichuan": conv_baichuan,
    "internlm": conv_internlm,
    "yiyan": conv_chatgpt,
    "xinghuo": conv_chatgpt,
    "qianwen": conv_chatgpt,
    "glm4v":conv_chatgpt,
    "xinghuo_vision": conv_chatgpt,
    "gemini": conv_chatgpt,
    "intern": conv_chatgpt,
    "huatuo-vision":conv_chatgpt
}

conv_templates_zh = {
    "llama": conv_llama2_zh,
    "baichuan": conv_baichuan_zh,
    "internlm": conv_internlm_zh,
    "chatgpt": conv_chatgpt_zh,
    "gpt4": conv_gpt4_zh,
    "yiyan": conv_chatgpt_zh,
    "xinghuo": conv_chatgpt_zh,
    "qianwen": conv_chatgpt_zh,
    "huatuo": conv_baichuan_zh,
    "chatglm3": conv_chatglm3_zh,
    "bloomz": conv_bloomz_zh,
    "glm4v": conv_chatglm3_zh,
    "xinghuo_vision": conv_chatgpt_zh
}

prompt_templates = {
    "base_v1_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n",
    "base_v1_zh": "你是一名经验丰富的[医生]，接下来有患者会来咨询病情，你需要对其进行诊断和治疗：\n\
<问诊要求>：\n\
        (1) 信息不充足的时候不要做出诊断，需要主动的向病人提问来获取信息。\n\
        (2) 每次提问时只能询问一个信息，话语尽可能的短。\n\
        (3) 建议对患者进行医学检查防止误判，提出检查后等待结果反馈进行下一步。\n\
        (4)  <\\image图片1>表明这是一张已上传图片，图片1是图片名字。\n\
        (5) 如果检查项目无法执行，请利用作用相同的其他检查项目进行代替。\n\
        (6) 当认为信息足够可以进行诊断和治疗的时候，请进行诊断并且输出：<结束问诊>。\n\n",
    "base_v2_zh": "你是一名经验丰富的医生，接下来有患者会来咨询病情，该问诊对话的长度最多为10轮。为了更好地诊断患者，你会根据每个回合的提示向患者提问。一旦你觉得已经获得了关于病人的足够信息，你就会给出诊断。\n \
<提示>：\n\
    1.会诊一般按照以下顺序进行：\n\
        a） 主诉，如主要症状和持续时间等\n\
        b） 基本信息，如年龄和性别等\n\
        c） 症状的原因，如旅行史和既往病史等\n\
        d） 医学检查，如诊断成像和生命体征。\n\
    2.每次问诊要尽可能短。\n\
    3.每次提问时只能询问一个信息。\n\
    4.当需要患者做检查的时候，请先输出<医学检查>，并且每次只能检查一个项目\n\
    5.当认为信息足够可以进行诊断和治疗的时候，请进行诊断并且给出治疗建议\n\n",
    "base_v2_en":"You are an experienced doctor, and a patient will soon consult you about their condition. This consultation will consist of up to 10 rounds of dialogue. To better diagnose the patient, you will ask questions based on prompts each round. Once you feel you have sufficient information about the patient, you will provide a diagnosis. \n\
    <Prompts>: \n\
    1.Consultations generally follow this order:\n\
        a) Chief complaint, such as main symptoms and duration.\n\
        b) Basic information, such as age and gender.\n\
        c) Possible causes of the symptoms, such as travel history and past medical history.\n\
        d) Medical examinations, such as X-ray and blood pressure.\n\
    2. The doctor's questions should be concise and clear, while the tone should be patient and caring for the patient.\n\
    3. The patient has already undergone all the necessary examinations for diagnosis, so the doctor can directly inquire about the results of the tests without requiring the patient to do further examinations.\n\
    4. There are only a maximum of 10 rounds of consultation dialogue, so the questions asked by the doctor in each round should help to determine the patient's most likely diagnosis or to clarify the next medical examination that should be done as much as possible.\n\n",
    "base_v2_en_cot":"You are an experienced doctor, and a patient will soon consult you about their condition. This consultation will consist of up to 10 rounds of dialogue. To better diagnose the patient, you will ask questions based on prompts each round. Once you feel you have sufficient information about the patient, you will provide a diagnosis. \n\
    <Prompts>: \n\
    1.Consultations generally follow this order:\n\
        a) Chief complaint, such as main symptoms and duration.\n\
        b) Basic information, such as age and gender.\n\
        c) Possible causes of the symptoms, such as travel history and past medical history.\n\
        d) Medical examinations, such as X-ray and blood pressure.\n\
    2. The doctor's questions should be concise and clear, while the tone should be patient and caring for the patient.\n\
    3. The patient has already undergone all the necessary examinations for diagnosis, so the doctor can directly inquire about the results of the tests without requiring the patient to do further examinations.\n\
    4. There are only a maximum of 10 rounds of consultation dialogue, so the questions asked by the doctor in each round should help to determine the patient's most likely diagnosis or to clarify the next medical examination that should be done as much as possible. \n\
    5. let's think step by step \n\
    <Case>:\n",
    "base_v2_en_zerocot":"You are an experienced doctor, and a patient will soon consult you about their condition. This consultation will consist of up to 10 rounds of dialogue. To better diagnose the patient, you will ask questions based on prompts each round. Once you feel you have sufficient information about the patient, you will provide a diagnosis. \n\
    <Prompts>: \n\
    1.Consultations generally follow this order:\n\
        a) Chief complaint, such as main symptoms and duration.\n\
        b) Basic information, such as age and gender.\n\
        c) Possible causes of the symptoms, such as travel history and past medical history.\n\
        d) Medical examinations, such as X-ray and blood pressure.\n\
    2. The doctor's questions should be concise and clear, while the tone should be patient and caring for the patient.\n\
    3. The patient has already undergone all the necessary examinations for diagnosis, so the doctor can directly inquire about the results of the tests without requiring the patient to do further examinations.\n\
    4. There are only a maximum of 10 rounds of consultation dialogue, so the questions asked by the doctor in each round should help to determine the patient's most likely diagnosis or to clarify the next medical examination that should be done as much as possible. \n\
    5. let's think step by step \n\n"
    # 2.Each sentence should be as brief as possible.\n\
    # 3.Only one piece of information can be inquired about per question.\n\
    # 4.When you need the patient to have an examination, please output <examination> first.\n\
    # 5.When the information is considered sufficient for diagnosis and treatment, please proceed with the diagnosis and provide treatment recommendations."
}

def get_doctor_template(mode, model_name):
    model_name = model_name.lower()

    if mode == "medqa":
        return get_value(conv_templates, model_name)
    else:
        return get_value(conv_templates_zh, model_name)

def get_doctor_prompt(prompt_id):
    prompt_id = prompt_id.lower()

    # if model_name in conv_templates.keys():
    return prompt_templates[prompt_id]

