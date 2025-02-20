import gradio as gr
import torch
from flux_context_pipeline import FluxContextImg2ImgPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import os

# current directory + /models
cache_path = os.path.join(os.getcwd(), "models")
os.environ["HF_HOME"] = cache_path
print(cache_path)

def create_prompt(model_prompt, cloth_prompt):
    prompt = f"The pair of images highlights a garment and its styling on a model; "
    prompt += f"[IMAGE1] {cloth_prompt};"
    prompt += f"[IMAGE2] {model_prompt};"
    return prompt

def preprocess_image(image, target_size=(768, 1024)):
    image = np.array(image)
    target_ratio = target_size[1] / target_size[0]
    current_ratio = image.shape[0] / image.shape[1]

    if current_ratio < target_ratio:
        # vertical padding
        pad_height = int(image.shape[1] * target_ratio - image.shape[0])
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        image = np.pad(image, ((pad_top, pad_bottom), (0, 0), (0, 0)), 
                      mode='constant', constant_values=255)
    else:
        # horizontal padding
        pad_width = int(image.shape[0] / target_ratio - image.shape[1])
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        image = np.pad(image, ((0, 0), (pad_left, pad_right), (0, 0)), 
                      mode='constant', constant_values=255)

    return Image.fromarray(image).resize(target_size)

pipe = None

def load_model(model_path):
    global pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16

    transformer = FluxTransformer2DModel.from_pretrained(
            model_path, 
            #subfolder="transformer", 
            torch_dtype=weight_dtype,
            cache_dir=cache_path
        )
    pipe = FluxContextImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=weight_dtype,
            cache_dir=cache_path
        ).to(device)
    
    print(" Model loaded from ", model_path)


def load_loras(lora_path1, lora_path2, lora_weight1=1.0, lora_weight2=1.0):
    # Descargar loras
    pipe.unload_lora_weights()

    # Cargarlos si se especificaron
    if lora_path1 != "":
        pipe.load_lora_weights(lora_path1, adapter_name = "lora1")
        print(" Lora weights loaded from ", lora_path1)
    if lora_path2 != "":
        pipe.load_lora_weights(lora_path2, adapter_name = "lora2")
        print(" Lora weights loaded from ", lora_path2)
    
    # Setear sus pesos
    if lora_path1!="" and lora_path2!="":
        pipe.set_adapters(["lora1", "lora2"], adapter_weights=[lora_weight1, lora_weight2])
    

def virtual_tryon(cloth_image, cloth_prompt, model_prompt, guidance_scale, num_inference_steps, lora_scale):
    # Constants
    size = (768, 1024)
            
    # Preprocess the input image
    processed_cloth = preprocess_image(cloth_image, size)
    
    # model input
    cloth_array = np.array(processed_cloth)
    model_input = np.column_stack([cloth_array, np.ones_like(cloth_array)*255])
    model_input = Image.fromarray(model_input)
    
    prompt_detailed = create_prompt(model_prompt, cloth_prompt)
    
    # Generate image
    result = pipe(
        prompt=prompt_detailed,
        prompt_2=prompt_detailed,
        height=size[1],
        width=size[0]*2,
        image=model_input,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=1.0,
        joint_attention_kwargs={"scale": lora_scale}
    ).images[0]
    
    return result

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# txt2img Virtual Try-On")
    
    # Load base model
    with gr.Row():
        model_path = gr.Textbox(label="Model Path", placeholder="")
        with gr.Column():
            load_model_btn = gr.Button("Load Model")
            result = gr.Textbox(label="Model Status", placeholder="Model not loaded")
    
    # Load LORAS
    with gr.Row():
        lora_path1 = gr.Textbox(label="Lora Path 1", placeholder="")
        lora_path2 = gr.Textbox(label="Lora Path 2", placeholder="")
        lora_weight1 = gr.Slider(
            minimum=0.1, maximum=1.3, value=1.0, step=0.05,
            label="Lora Weight 1"
        )
        lora_weight2 = gr.Slider(
            minimum=0.1, maximum=1.3, value=1.0, step=0.05,
            label="Lora Weight 2"
        )
        result_lora = gr.Textbox(label="Lora Status", placeholder="Loras not loaded")
        load_loras_btn = gr.Button("Load Loras")
    

    load_model_btn.click(
        fn=load_model,
        inputs=[model_path],
        outputs=result
    )

    load_loras_btn.click(
        fn=load_loras,
        inputs =[lora_path1, lora_path2,lora_weight1, lora_weight2],
        outputs = result_lora
    )

    with gr.Row():
        with gr.Column():
            cloth_image = gr.Image(label="Upload Garment Image")
            cloth_prompt = gr.Textbox(label="Cloth Description", placeholder="e.g., Cargo pants")
            model_prompt = gr.Textbox(
                label="Model Description", 
                placeholder="e.g., Man in front of a clean white background"
            )
            guidance_scale = gr.Slider(
                minimum=1, maximum=20, value=4, step=0.5,
                label="Guidance Scale"
            )
            num_inference_steps = gr.Slider(
                minimum=1, maximum=100, value=20, step=1,
                label="Number of Inference Steps"
            )
            lora_scale = gr.Slider(
                minimum=0.1, maximum=1.3, value=1.0, step=0.05,
                label="Lora Scale"
            )
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Result")
    
    generate_btn.click(
        fn=virtual_tryon,
        inputs=[
            cloth_image,
            cloth_prompt,
            model_prompt,
            guidance_scale,
            num_inference_steps,
            lora_scale
        ],
        outputs=output_image
    )
    

if __name__ == "__main__":
    demo.launch(share=True)