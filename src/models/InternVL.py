from .base_model import API_Model, Local_Model, LOCAL_MODEL_PATHS, KeywordsStoppingCriteria, Base_Model
import pdb
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from utils.general_utils import disable_torch_init

import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL_Model(Base_Model):
    def __init__(self, 
                 model_name="Mini-InternVL-Chat-4B-V1-5",
                 stop_ids=[]):
        super().__init__()
        model_path = LOCAL_MODEL_PATHS[model_name]
        self.model_path = os.path.expanduser(model_path)
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        if model_name == 'InternVL-Chat-V1-5':
            self.model = AutoModelForCausalLM.from_pretrained(model_path,  
                                                          torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True,
                                                          trust_remote_code=True,
                                                          device_map="auto").eval().cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path,  
                                                          torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True,
                                                          trust_remote_code=True).eval().cuda()
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        stop_ids.append(torch.tensor([self.tokenizer.eos_token_id])) # </s>
        # pdb.set_trace()
        # stop_ids.append(torch.tensor([self.tokenizer("PATIENT")]))
        self.stop_ids = [s.cuda() for s in stop_ids]
        self.stop_criteria = KeywordsStoppingCriteria(self.stop_ids)
    
    def generate(self, inputs, images=None):

        content_images = []
        if images != [] and images is not None:
            for image in images:
                pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()
                content_images.append(pixel_values)
            pixel_values = torch.stack(content_images).squeeze().view(-1,pixel_values.size(1),pixel_values.size(2),pixel_values.size(3))
        else:
            pixel_values  = None

        while True:
            try:
                response, history = self.model.chat(self.tokenizer, pixel_values, inputs, self.generation_config, history=None, return_history=True)

                outputs = response
                break
            except:
                print("Error, Retrying...")
                import pdb
                pdb.set_trace()

        return outputs
