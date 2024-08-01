import os
import pathlib
import google.generativeai as genai
from .base_model import API_Model
import pdb

class Gemini_Vision_Model(API_Model):
    def __init__(self, 
                 api_key="",
                 model_name = 'gemini-1.5-pro',
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        self.api_key = []
        self.model_name = model_name
        self.key_tim = 0
        # genai.configure(api_key = api_key)
        # self.model = genai.GenerativeModel(model_name,
        #                                 generation_config=genai.GenerationConfig(
        #                                 max_output_tokens=1000,
        #                                 temperature=0.01,
        #                                 ))
    
    def generate(self, inputs, images=None):

        content = [inputs]
        if images is not None:
            for image in images:
                content.append({
                    'mime_type': 'image/jpeg',
                    'data': pathlib.Path(image).read_bytes()
                })
        message = content

        while True:
            try:
                genai.configure(api_key = self.api_key[self.key_tim])
                model = genai.GenerativeModel(self.model_name,
                                                generation_config=genai.GenerationConfig(
                                                max_output_tokens=1000,
                                                temperature=0.01,
                                                ))
                self.key_tim = 3 - self.key_tim
                
                response = model.generate_content(message)
                outputs = response.text
                break
            except:
                print("Error, Retrying...")
                # pdb.set_trace()

        return outputs
    