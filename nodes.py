import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from .src.pix2pix_turbo import Pix2Pix_Turbo
from .src.image_prep import canny_from_pil
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
loras_path = folder_paths.get_folder_paths("loras")[0]

class Img2ImgTurboSketchLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

            },
        }

    RETURN_TYPES = ("Img2ImgTurboSketchModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Img2ImgTurbo"

    def run(self):
        model_name="sketch_to_image_stochastic"
        model = Pix2Pix_Turbo(pretrained_name=model_name,ckpt_folder=loras_path)
        model.set_eval()
        return (model,)

class Img2ImgTurboEdgeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("Img2ImgTurboEdgeModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Img2ImgTurbo"

    def run(self):
        model_name="edge_to_image"
        model = Pix2Pix_Turbo(pretrained_name=model_name,ckpt_folder=loras_path)
        model.set_eval()
        return (model,)

class Img2ImgTurboSketchRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("Img2ImgTurboSketchModel",),
                "image": ("IMAGE",),
                "prompt": ("STRING",{"default":""}),
                "seed": ("INT",{"default":1234}),
                "gamma": ("FLOAT",{"default":0.4}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Img2ImgTurbo"

    def run(self,model,image,prompt,seed,gamma):
        low_threshold=100
        
        image = 255.0 * image[0].cpu().numpy()
        input_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        image_t = F.to_tensor(input_image) < 0.5
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        output_image = model(c_t, prompt, deterministic=False, r=gamma, noise_map=noise)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        return (torch.unsqueeze(torch.tensor(np.array(output_pil).astype(np.float32) / 255.0), 0),)

class Img2ImgTurboEdgeRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("Img2ImgTurboEdgeModel",),
                "image": ("IMAGE",),
                "prompt": ("STRING",{"default":""}),
                "seed": ("INT",{"default":1234}),
                "gamma": ("FLOAT",{"default":0.4}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Img2ImgTurbo"

    def run(self,model,image,prompt,seed,gamma):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')

        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)

        #image = image[:, :, None]
        #image = np.concatenate([image, image, image], axis=2)
        #image = Image.fromarray(image)
        c_t = F.to_tensor(image).unsqueeze(0).cuda()
        output_image = model(c_t, prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        return (torch.unsqueeze(torch.tensor(np.array(output_pil).astype(np.float32) / 255.0), 0),)

NODE_CLASS_MAPPINGS = {
    "Img2ImgTurboSketchLoader":Img2ImgTurboSketchLoader,
    "Img2ImgTurboEdgeLoader":Img2ImgTurboEdgeLoader,
    "Img2ImgTurboSketchRun":Img2ImgTurboSketchRun,
    "Img2ImgTurboEdgeRun":Img2ImgTurboEdgeRun,
}