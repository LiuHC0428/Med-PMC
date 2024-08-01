from zhipuai import ZhipuAI
from .base_model import API_Model, Local_Model, LOCAL_MODEL_PATHS, KeywordsStoppingCriteria, Base_Model
import base64
import pdb



class GLM_Vision_Model(API_Model):
    def __init__(self, 
                 api_key="",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        self.client = ZhipuAI(api_key=api_key)
    
    def generate(self, inputs, images=None):

        content = [{'type':'text','text':inputs}]
        if images is not None:
            for image in images:
                with open(image,'rb') as f:
                    image_data = base64.b64encode(f.read())
                content.append({"type": "image_url",'image_url':{'url':str(image_data, encoding='utf-8')}})
        message = [{"role": "user", "content": content}]

        while True:
            try:
                response = self.client.chat.completions.create(model="glm-4v", messages=message,temperature=0.01, seed=0)

                outputs = response.choices[0].message.content

                break
            except:
                # pdb.set_trace()
                print("Error, Retrying...")

        return outputs
    