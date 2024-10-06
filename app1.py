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
loaded_ipadapter = None


def load_models(checkpoint_name, lora_name):
    global loaded_checkpoint, loaded_lora, loaded_ipadapter
    
    with torch.inference_mode():
        if loaded_checkpoint is None:
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            loaded_checkpoint = checkpointloadersimple.load_checkpoint(
                ckpt_name=checkpoint_name
            )
        
        if loaded_lora is None:
            loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            loaded_lora = loraloadermodelonly.load_lora_model_only(
                lora_name=lora_name,
                strength_model=1,
                model=get_value_at_index(loaded_checkpoint, 0),
            )
        
        if loaded_ipadapter is None:
            ipadapterunifiedloaderfaceid = NODE_CLASS_MAPPINGS[
                "IPAdapterUnifiedLoaderFaceID"
            ]()
            loaded_ipadapter = ipadapterunifiedloaderfaceid.load_models(
                preset="FACEID PLUS V2",
                lora_strength=1,
                provider="CUDA",
                model=get_value_at_index(loaded_lora, 0),
            )


def run_workflow(user_prompt, negative_prompt, input_image_path):
    with torch.inference_mode():
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_5 = emptylatentimage.generate(width=512, height=768, batch_size=1)

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=user_prompt,
            clip=get_value_at_index(loaded_checkpoint, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(loaded_checkpoint, 1),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_27 = loadimage.load_image(image=input_image_path)

        ipadapterfaceid = NODE_CLASS_MAPPINGS["IPAdapterFaceID"]()
        ipadapterfaceid_18 = ipadapterfaceid.apply_ipadapter(
            weight=1,
            weight_faceidv2=1.25,
            weight_type="linear",
            combine_embeds="concat",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(loaded_ipadapter, 0),
            ipadapter=get_value_at_index(loaded_ipadapter, 1),
            image=get_value_at_index(loadimage_27, 0),
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        ksampler_3 = ksampler.sample(
            seed=2,
            steps=1,
            cfg=1,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(ipadapterfaceid_18, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(loaded_checkpoint, 2),
        )

        tensor_image = get_value_at_index(vaedecode_8, 0)
        numpy_image = tensor_image.squeeze().numpy()
        numpy_image = (numpy_image * 255).astype(np.uint8)

        pil_image = Image.fromarray(numpy_image)
        img_io = BytesIO()
        pil_image.save(img_io, "PNG")
        img_io.seek(0)
        
        return img_io


@app.route("/", methods=["GET", "POST"])
def index():
    generated_image_b64 = None
    if request.method == "POST":
        user_prompt = request.form["prompt"]
        negative_prompt = request.form.get(
            "negative_prompt",
            "blurry, noisy, lowres, messy, jpeg, artifacts, ill, distorted, malformed, naked",
        )
        checkpoint_option = request.form["checkpoint"]
        lora_option = request.form["lora"]
        image_source = request.form["image_source"]

        print(f"Image source: {image_source}")

        # Use BASE_DIR to create an absolute path
        image_save_path = os.path.join(BASE_DIR, "input", "image", "input_image.jpg")
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)

        if image_source == "upload":
            uploaded_image = request.files["upload"]
            if uploaded_image:
                uploaded_image.save(image_save_path)
                print("Uploaded image saved")
            else:
                print("No uploaded image found")
        elif image_source == "webcam":
            captured_image = request.form["captured_image"]
            if captured_image:
                # Remove the "data:image/jpeg;base64," prefix
                image_data = captured_image.split(",")[1]
                with open(image_save_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                print("Webcam image saved")
            else:
                print("No webcam image data received")

        if os.path.exists(image_save_path):
            print(f"Input image saved to: {image_save_path}")
            load_models(checkpoint_option, lora_option)
            generated_image = run_workflow(user_prompt, negative_prompt, image_save_path)

            # Convert the image to base64 for embedding in HTML
            generated_image.seek(0)
            generated_image_b64 = base64.b64encode(generated_image.getvalue()).decode(
                "utf-8"
            )
        else:
            print("No input image found")

    return render_template("index.html", generated_image=generated_image_b64)


if __name__ == "__main__":
    app.run(debug=False)