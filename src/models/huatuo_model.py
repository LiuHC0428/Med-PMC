from .HuatuoGPT_Vision.cli import HuatuoChatbot
from .base_model import API_Model, Local_Model, LOCAL_MODEL_PATHS, KeywordsStoppingCriteria, Base_Model

class Huatuo_Vision_Model(Base_Model):
    def __init__(self, 
                 model_name="huatuo_vision-7b",
                 stop_ids=[]):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name]
        # self.model_path = os.path.expanduser(model_path)
        self.model = HuatuoChatbot(model_path)
    
    def generate(self, inputs, images=None):

        content_images = []
        if images != [] and images is not None:
            for image in images:
                content_images.append(image)
        else:
            pixel_values  = None

        while True:
            try:
                if content_images != []:
                    response = self.model.inference(inputs, content_images)[0]
                else:
                    response = self.model.inference(inputs)[0]
                outputs = response
                break
            except:
                print("Error, Retrying...")
                import pdb
                pdb.set_trace()

        return outputs
