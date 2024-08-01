from .base_model import *
from .openai_model import *
from .qianwen_model import *
from .xinghuo_model import *
from .yiyan_model import *
from .GLM4_model import *
from .gimini_pro import *
from .InternVL import *
from .huatuo_model import *

def get_model(model_name, stop_ids=[]):
    # api model
    if model_name == "chatgpt":
        return OpenAI_Model(model_type="gpt-3.5-turbo-1106", stop_ids=stop_ids)
    elif model_name == "gpt4":
        return OpenAI_Model(model_type="gpt-4-1106-preview", stop_ids=stop_ids)
    elif model_name == "qianwen":
        return QianWen_Model(stop_ids=stop_ids)
    elif model_name == "qianwen-vision":
        return QianWen_Vision_Model(stop_ids=stop_ids)
    elif model_name == 'qianwen-vision-chat':
        return QianWen_Vision_Local_Model(model_name ='qwen-vl-chat')
    elif model_name == "xinghuo":
        return XingHuo_Model(stop_ids=stop_ids)
    elif model_name == "yiyan":
        return YiYan_Model(stop_ids=stop_ids)
    elif model_name == "glm4v":
        return GLM_Vision_Model(stop_ids=stop_ids)
    elif model_name == "xinghuo_vision":
        return XingHuo_Vision_Model(stop_ids=stop_ids)
    elif model_name == "gpt4o":
        return OpenAI_VISION_Model(model_type="gpt-4o-2024-05-13", stop_ids=stop_ids)
    elif model_name == "mini-gpt4o":
        return OpenAI_VISION_Model(model_type="gpt-4o-mini", stop_ids=stop_ids)
    elif model_name == "gpt4v":
        return OpenAI_VISION_Model(model_type="gpt-4-turbo-2024-04-09", stop_ids=stop_ids)
    elif model_name == 'gemini-pro':
        return Gemini_Vision_Model(model_name = 'gemini-1.5-pro')
    elif model_name == 'gemini-flash':
        return Gemini_Vision_Model(model_name = 'gemini-1.5-flash')
    elif model_name == 'InternVL-1.5':
        return InternVL_Model(model_name = 'InternVL-Chat-V1-5')
    elif model_name == 'Mini-InternVL-1.5':
        return InternVL_Model(model_name = 'Mini-InternVL-Chat-4B-V1-5')
    elif model_name == 'huatuo-vision-7b':
        return Huatuo_Vision_Model(model_name = 'huatuo-vision-7b')
    elif model_name == 'huatuo-vision-34b':
        return Huatuo_Vision_Model(model_name = 'huatuo-vision-34b')
    else:
        return Local_Model(model_name, stop_ids)
    