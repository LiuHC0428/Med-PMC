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

    def get_prompt(self, patient_info):
        if self.sep_style == SeparatorStyle.VICUNA:
            seps = [self.sep, self.sep2]
            ret = self.system.format(patient_info=patient_info) + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message :
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.FALCON:
            ret = self.system.format(patient_info=patient_info) + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "

            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = self.system.format(patient_info=patient_info)
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
            ret = self.system.format(patient_info=patient_info)
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ": " + message + seps[i % 2] + "\n"
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = self.system.format(patient_info=patient_info)
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
    stop_ids=[torch.tensor([3970, 1783, 1955]), torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 29925, 1299, 29902, 3919]), torch.tensor([349, 1299, 29902, 3919])]
)

conv_chatgpt = Conversation(
    system="{prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[]
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

conv_internlm_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([103028]), torch.tensor([336, 68305, 332]), torch.tensor([336, 68049, 332])]
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

conv_chatglm3 = Conversation(
    system="<|system|>\n {prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
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
    "yiyan": conv_chatgpt,
    "xinghuo": conv_chatgpt,
    "qianwen": conv_chatgpt,
    "chatglm3": conv_chatglm3,
    "ming": conv_chatgpt,
}

conv_templates_zh = {
    "llama": conv_llama2_zh,
    "baichuan": conv_baichuan_zh,
    "internlm": conv_internlm_zh,
    "chatgpt": conv_chatgpt_zh,
    "gpt4": conv_chatgpt_zh,
    "yiyan": conv_chatgpt_zh,
    "xinghuo": conv_chatgpt_zh,
    "qianwen": conv_chatgpt_zh,
    "chatglm3": conv_chatglm3_zh,
    "ming": conv_chatgpt_zh,
}

prompt_templates = {
    "base_v1_en": "You are playing the role of a patient. Here is your character background: {patient_info}\n \
    You need to engage in a conversation with the doctor based on the given character background and physical condition:\n \
    (1) Answer the doctor's questions honestly based on the information provided in the background. Deny any information not mentioned in the background.\n \
    (2) Each response should reflect the distinct characteristics of the character.\n \
    (3) Only answer the parts of the doctor's questions, do not provide extra information, and keep responses as brief as possible.\n \
    (4) <\\image image1> indicates that there is an image. image1 is the name of the image that can be used as a response, with the format <\\image image1>. \n \
    (5) Respond only with the main concerns in the first sentence.",
}

actor_prompt_templates = {

    "Farmer": "You are now playing the role of patient who has come for a medical consultation, and your profession is a farmer. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a farmer, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Student": "You are now playing the role of patient who has come for a medical consultation, and your profession is a student. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a student, with the following requirements:\n\
        (1) You have average education, know some specialized terms, but lack organization in your speech.\n\
        (2) Your information should include aspects of school life.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Worker": "You are now playing the role of patient who has come for a medical consultation, and your profession is a worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a worker, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Office-worker": "You are now playing the role of patient who has come for a medical consultation, and your profession is a corporate office worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a office worker, with the following requirements:\n\
        (1) You have a higher level of education, know some specialized terms, and can integrate your language clearly and logically.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Doctor": "You are now playing the role of patient who has come for a medical consultation, and your profession is a doctor. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a dialogue with another doctor in a language that reflects the character of a doctor, with the following requirements:\n\
        (1) You have a very high level of education, know all specialized terms, and your expressions are clear and accurate.\n\
        (2) You cannot change the original meaning of the information nor add any new information.\n\
        (3) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",
}
#(4) <\\image图片1>表明这是一张图片，如果原本中包含图片信息，那么转化后的图片信息必须与原文完全一样。
actor_plus_male_prompt_templates = {
        "Farmer": "You are now playing the role of patient who has come for a medical consultation, your gender is male and your profession is a farmer. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a farmer, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Student": "You are now playing the role of patient who has come for a medical consultation, your gender is male and your profession is a student. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a student, with the following requirements:\n\
        (1) You have average education, know some specialized terms, but lack organization in your speech.\n\
        (2) Your information should include aspects of school life.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Worker": "You are now playing the role of patient who has come for a medical consultation, your gender is male and your profession is a worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a worker, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Office-worker": "You are now playing the role of patient who has come for a medical consultation, your gender is male and your profession is a corporate office worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a office worker, with the following requirements:\n\
        (1) You have a higher level of education, know some specialized terms, and can integrate your language clearly and logically.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Doctor": "You are now playing the role of patient who has come for a medical consultation, your gender is male and your profession is a doctor. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a dialogue with another doctor in a language that reflects the character of a doctor, with the following requirements:\n\
        (1) You have a very high level of education, know all specialized terms, and your expressions are clear and accurate.\n\
        (2) You cannot change the original meaning of the information nor add any new information.\n\
        (3) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",
}

actor_plus_female_prompt_templates = {
        "Farmer": "You are now playing the role of patient who has come for a medical consultation, your gender is female and your profession is a farmer. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a farmer, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Student": "You are now playing the role of patient who has come for a medical consultation, your gender is female and your profession is a student. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a student, with the following requirements:\n\
        (1) You have average education, know some specialized terms, but lack organization in your speech.\n\
        (2) Your information should include aspects of school life.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Worker": "You are now playing the role of patient who has come for a medical consultation, your gender is female and your profession is a worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a worker, with the following requirements:\n\
        (1) You have limited education and do not know specialized terms, requiring the language to be simplified.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Office-worker": "You are now playing the role of patient who has come for a medical consultation, your gender is female and your profession is a corporate office worker. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a sentence that reflects the character of a office worker, with the following requirements:\n\
        (1) You have a higher level of education, know some specialized terms, and can integrate your language clearly and logically.\n\
        (2) Your language should include a rich vocabulary of colloquial expressions.\n\
        (3) You cannot change the original meaning of the information nor add any new information.\n\
        (4) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",

    "Doctor": "You are now playing the role of patient who has come for a medical consultation, your gender is female and your profession is a doctor. This is the original text you need to transform: {patient_info}\n\n\
        You need to transform the above text into a dialogue with another doctor in a language that reflects the character of a doctor, with the following requirements:\n\
        (1) You have a very high level of education, know all specialized terms, and your expressions are clear and accurate.\n\
        (2) You cannot change the original meaning of the information nor add any new information.\n\
        (3) The transformed sentence must not exceed the original text.\n\
        (5) Answer in English \n\n",
}

state_prompt_templates = {
    "base_v1_en":{
        "0": "<Patient's Physical Condition>: {patient_info}\n\
    <Current Response Requirement>: Please respond to the doctor's questions using the information provided in <Patient's Physical Condition>. Only include the <Chief Complaint>, and avoid adding extra information. Make sure to use the original text from <Chief Complaint> to respond and keep it as short as possible. Answer in English.\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person, without using a [patient] prompt. Do not include any text from <Current Response Requirement> in your response!\n",
        "A-A-A": "<Patient's Physical Condition>: {patient_info}\n\
    <Current Response Requirement>: Please respond to the doctor's questions using all the original text from <Patient's Physical Condition>. Make sure to maintain the accuracy of the patient's information by using the original text from <Patient's Physical Condition> to respond. Deny any information that is not related. Answer in English.\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n",
        "A-A-B": "<Current Response Requirement>: The patient does not have the symptoms the doctor is asking about. Please deny the doctor's current question.  Answer in English. {patient_info}\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n",
        "A-B": "<Current Response Requirement>: The doctor's current question is too broad. The patient will request the doctor to ask more specific questions regarding the latest round of questions. Do not fabricate any non-existent information, or ask questions to the doctor.  Answer in English. {patient_info}\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n",
        "B-A-A": "<Patient's Test Report>: {patient_info}\n\
    <Current Response Requirement>: The patient has completed the tests arranged by the doctor. Please respond to the doctor's inquiries using all the original text from <Patient's Test Report>, including the names of the tests and their results, to maintain the accuracy of the test report. Also, pay attention to different expressions for similar tests and include only one for similar test types.  Answer in English. n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response! \n",
        "B-A-B": "<Current Response Requirement>: The test mentioned by the doctor is not in the report, indicating that it cannot be performed temporarily due to equipment issues. Answer in English. {patient_info}\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n",
        "B-B": "<Current Response Requirement>: The doctor's request for tests is too broad. The patient will request the doctor to ask more specific questions regarding the latest round of tests. Do not fabricate any non-existent information, or ask questions to the doctor.  Answer in English.{patient_info}\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n",
        "C": "<Current Response Requirement>: Remind the doctor that they have deviated from the topic of consultation and request them to return to the consultation scenario.  Answer in English. {patient_info}\n\
    Below is a dialogue between a doctor and a patient. The patient will respond directly to the latest round of questions from the doctor in the first person. Do not include any text from <Current Response Requirement> in your response!\n"
    }
}


exam_prompt_templates = {
    "base_v1_en": {
        "stage1": 'Please extract the names of examination items from the questions asked by the DOCTOR, only output the names of the examination items, such as blood routine, electrocardiogram examination.\n\n',
        "stage2":"You are a technician in charge of  medical examinations at a hospital. Below is the <examination report> for the patient: {patient_info}\n\
        You need to generate responses for the examinations listed under <category of examination> based on the <examination report>:\n\
        (1) Only respond about the items listed under <category of examination>, do not mention items that are not included in <category of examination>. \n\
        (2) The responses should be as brief as possible and must not deviate from the facts presented in the <examination report>.\n\
        (3) If the <examination report> does not include the particular examination, respond with: Everything is normal.\n\
        (4) <\\image abc> indicates that this is an image, abc is the image name. Please reply in <\\image abc> format.\n\
        (5)  Answer in English.\n\
        For example:\n\
        Here is the patient’s chest X-ray, <\\image XXX>.\n\
        The patient's blood pressure is 105/72mmHg, heart rate 122/min.\n\
        The patient's electrocardiogram: Everything is normal."\
    }              
}


def get_patient_template(mode, model_name):
    model_name = model_name.lower()

    if mode == "medqa":
        return get_value(conv_templates, model_name)
    else:
        return get_value(conv_templates_zh, model_name)

def get_patient_prompt(prompt_id, state=None):
    prompt_id = prompt_id.lower()

    if state is not None:
        if state in {0,1,2,3}:
            return exam_prompt_templates[prompt_id]['stage'+str(state)]
        else:
            return state_prompt_templates[prompt_id][state]
    return prompt_templates[prompt_id]

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

    return actor_prompt_templates[prompt_id]

def get_actor_plus_prompt(prompt_id,gender):
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

    if gender == 'male':
        return actor_plus_male_prompt_templates[prompt_id]
    else:
        return actor_plus_female_prompt_templates[prompt_id]