from flux_context_pipeline import FluxContextImg2ImgPipeline
from diffusers import FluxTransformer2DModel
import torch
from diffusers.utils import load_image

def create_prompt(model_prompt, cloth_prompt):
    prompt = f"The pair of images highlights a garment and its styling on a model; "
    prompt += f"[IMAGE1] {cloth_prompt};"
    prompt += f"[IMAGE2] {model_prompt};"
    return prompt

prompt_generic = f"The pair of images highlights a garment and its styling on a model; [IMAGE1] Detailed product shot of a garment; [IMAGE2] The same cloth is worn by a model;"
                
device = "cuda:6"
transformer = FluxTransformer2DModel.from_pretrained(
    "/workspace1/pdawson/tryon-finetune/trained-flux-txt2img/checkpoint-6000", subfolder="transformer"
)
weight_dtype = torch.bfloat16

pipe = FluxContextImg2ImgPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",  # model
                        transformer = transformer,
                        torch_dtype=weight_dtype).to(device)

print("pipe loaded fine bro")

test_cloth = load_image("/workspace1/pdawson/tryon-scraping/dataset/test/cloth/d80d5841899a4eb8850a8a7a6df901de32cf51c5.jpg")

cloth_prompt = ""
model_prompt = ""
prompt_detailed = create_prompt(model_prompt, cloth_prompt)

