from share import *
import config
config.save_memory = True

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

print("prepare CannyDetector ...")
apply_canny = CannyDetector()

input_image = cv2.imread('d:/data/image/test.jpg')
cv2.imshow('test input', input_image)
cv2.waitKey(10)

print("create cldm model...")
model = create_model('./models/cldm_v15.yaml').cpu()
print("load weights ...")
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))
print("move model to cuda")
model = model.cpu()
print("DDIMSampler")
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
image_resolution = 256#gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
strength = 1.0 #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
low_threshold = 100#gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
high_threshold = 200#gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
ddim_steps = 10 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
scale = 9.0#gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
seed = 12345#gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
eta = 0.0#gr.Number(label="eta (DDIM)", value=0.0)
a_prompt = 'best quality, extremely detailed'#gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers'#gr.Textbox(label="Negative Prompt",


prompt = "a hammer on a soft carpet"

print("process: start: prompt", prompt)
res = process(input_image, prompt, a_prompt, n_prompt, 
        num_samples, image_resolution, ddim_steps, 
        guess_mode, strength, scale, seed, eta, 
        low_threshold, high_threshold)


cv2.imwrite('output1.png', res[0])
cv2.imwrite('output2.png', res[1])
cv2.imshow('control1', res[0])
cv2.imshow('control2', res[1])
cv2.waitKey(0)



