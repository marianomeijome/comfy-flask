
from multiprocessing.connection import wait
import streamlit as st
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from nodes import NODE_CLASS_MAPPINGS


# Streamlit App

st.title("GenAI Workflow with Streamlit")


#test strealit webcam widget
picture = st.camera_input("Take a picture")
webcam_save_path = '/webcam/webcam.jpg'

if picture:
    with open ('input'+webcam_save_path,'wb') as file:
        file.write(picture.getbuffer())

user_prompt = st.text_input("Prompt", "Elon musk smoking a joint")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
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
            return  # Exit early if no config file is found

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS
import_custom_nodes()

# Move these variables outside of any function
checkpoint_options = st.selectbox("Select a checkpoint", ("v1-5-pruned-emaonly-fp16.safetensors", "Realistic_Vision_V6.0_NV_B1_fp16.safetensors", "v1-5-pruned-emaonly.safetensors", "sd_xl_base_1.0.safetensors"))
lora_options = st.selectbox("Select a lora", ("Hyper-SD15-1step-lora.safetensors", "Hyper-SDXL-1step-lora.safetensors", "papi2.safetensors", "TCD_SD15_LoRA.safetensors", "TQ_-_Stained_Glass_XL-000006.safetensors"))

# Global variables to store loaded models
loaded_checkpoint = None
loaded_lora = None

def load_models():
    global loaded_checkpoint, loaded_lora
    
    with torch.inference_mode():
        if loaded_checkpoint is None:
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            loaded_checkpoint = checkpointloadersimple.load_checkpoint(ckpt_name=checkpoint_options)
        
        if loaded_lora is None:
            loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            loaded_lora = loraloadermodelonly.load_lora_model_only(
                lora_name=lora_options,
                strength_model=1,
                model=get_value_at_index(loaded_checkpoint, 0),
            )

with torch.inference_mode():
    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    checkpointloadersimple_14 = checkpointloadersimple.load_checkpoint(
        ckpt_name=checkpoint_options
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
        image="E:\ComfyUI\input\webcam\webcam.jpg"
    )

    vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    vaeencode_12 = vaeencode.encode(
        pixels=get_value_at_index(loadimage_34,0),
        vae=get_value_at_index(checkpointloadersimple_14, 2),
    )

    loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    loraloadermodelonly_17 = loraloadermodelonly.load_lora_model_only(
        lora_name=lora_options,
        strength_model=1,
        model=get_value_at_index(checkpointloadersimple_14, 0),
    )


def run_workflow():
    with torch.inference_mode():
        tcdmodelsamplingdiscrete = NODE_CLASS_MAPPINGS["TCDModelSamplingDiscrete"]()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
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

            saveimage_25 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            )

            tensor_image = get_value_at_index(vaedecode_8, 0)

            # Convert the tensor to a NumPy array
            numpy_image = tensor_image.squeeze().numpy()  # Remove unnecessary dimensions

            # Scale pixel values to [0, 255] and cast to uint8
            numpy_image = (numpy_image * 255).astype(np.uint8)

            # Display the image using st.image
            st.image(numpy_image, caption="Generated Image", clamp=True)

# Button to run the workflow
if st.button("Run Workflow"):       
    run_workflow()
