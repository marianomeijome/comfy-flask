from flask import Flask, render_template, request, send_file, url_for
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# At the top of your file, add this line to get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py.")
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            print("Could not import load_extra_path_config from utils.extra_config.")
            return

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def import_custom_nodes() -> None:
    from nodes import init_extra_nodes
    init_extra_nodes()

# Initialize ComfyUI components
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()

from nodes import NODE_CLASS_MAPPINGS

# Global variables to store loaded models
loaded_checkpoint = None
loaded_lora = None

def load_models(checkpoint_option, lora_option):
    global loaded_checkpoint, loaded_lora
    
    with torch.inference_mode():
        if loaded_checkpoint is None:
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            loaded_checkpoint = checkpointloadersimple.load_checkpoint(ckpt_name=checkpoint_option)
        
        if loaded_lora is None:
            loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            loaded_lora = loraloadermodelonly.load_lora_model_only(
                lora_name=lora_option,
                strength_model=1,
                model=get_value_at_index(loaded_checkpoint, 0),
            )

def run_workflow(user_prompt, checkpoint_option, lora_option, webcam_image_path):
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_14 = checkpointloadersimple.load_checkpoint(
            ckpt_name=checkpoint_option
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=user_prompt,
            clip=get_value_at_index(checkpointloadersimple_14, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="watermark, text\n",
            clip=get_value_at_index(checkpointloadersimple_14, 1),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_34 = loadimage.load_image(
            image=webcam_image_path
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_12 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_34,0),
            vae=get_value_at_index(checkpointloadersimple_14, 2),
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_17 = loraloadermodelonly.load_lora_model_only(
            lora_name=lora_option,
            strength_model=1,
            model=get_value_at_index(checkpointloadersimple_14, 0),
        )

        tcdmodelsamplingdiscrete = NODE_CLASS_MAPPINGS["TCDModelSamplingDiscrete"]()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        tcdmodelsamplingdiscrete_19 = tcdmodelsamplingdiscrete.patch(
            steps=1,
            scheduler="simple",
            denoise=0.5,
            eta=0.8,
            model=get_value_at_index(loraloadermodelonly_17, 0),
        )

        samplercustom_18 = samplercustom.sample(
            add_noise=True,
            noise_seed=random.randint(1, 2**64),
            cfg=1,
            model=get_value_at_index(tcdmodelsamplingdiscrete_19, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            sampler=get_value_at_index(tcdmodelsamplingdiscrete_19, 1),
            sigmas=get_value_at_index(tcdmodelsamplingdiscrete_19, 2),
            latent_image=get_value_at_index(vaeencode_12, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(samplercustom_18, 0),
            vae=get_value_at_index(checkpointloadersimple_14, 2),
        )

        tensor_image = get_value_at_index(vaedecode_8, 0)
        numpy_image = tensor_image.squeeze().numpy()
        numpy_image = (numpy_image * 255).astype(np.uint8)

        pil_image = Image.fromarray(numpy_image)
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return img_io

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_image_b64 = None
    if request.method == 'POST':
        user_prompt = request.form['prompt']
        checkpoint_option = request.form['checkpoint']
        lora_option = request.form['lora']
        webcam_image = request.files['webcam']

        if webcam_image:
            # Use BASE_DIR to create an absolute path
            webcam_save_path = os.path.join(BASE_DIR, 'input', 'webcam', 'webcam.jpg')
            os.makedirs(os.path.dirname(webcam_save_path), exist_ok=True)
            print(f"Saving webcam image to: {webcam_save_path}")
            webcam_image.save(webcam_save_path)

        load_models(checkpoint_option, lora_option)
        generated_image = run_workflow(user_prompt, checkpoint_option, lora_option, webcam_save_path)

        # Convert the image to base64 for embedding in HTML
        generated_image.seek(0)
        generated_image_b64 = base64.b64encode(generated_image.getvalue()).decode('utf-8')

    return render_template('index.html', generated_image=generated_image_b64)

if __name__ == '__main__':
    app.run(debug=True)