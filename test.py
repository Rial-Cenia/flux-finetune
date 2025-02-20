from flux_context_pipeline import FluxContextImg2ImgPipeline
from diffusers import FluxTransformer2DModel
import torch
from diffusers.utils import load_image
import numpy as np
from PIL import Image

def create_prompt(model_prompt, cloth_prompt):

    prompt = f"The pair of images highlights a garment and its styling on a model; "
    prompt += f"[IMAGE1] {cloth_prompt};"
    prompt += f"[IMAGE2] {model_prompt};"

    return prompt

prompt_generic = f"The pair of images highlights a garment and its styling on a model; [IMAGE1] Detailed product shot of a garment; [IMAGE2] The same cloth is worn by a model;"
                
device = "cuda:6"
weight_dtype = torch.bfloat16

transformer = FluxTransformer2DModel.from_pretrained(
    "/workspace1/pdawson/tryon-finetune/trained-flux-txt2img/checkpoint-15000", subfolder="transformer", torch_dtype=weight_dtype
)

# push to hubs
transformer.push_to_hub("RIAL-AI/txt2img-tryon-15k")

pipe = FluxContextImg2ImgPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",  # model
                        transformer = transformer,
                        torch_dtype=weight_dtype).to(device)

test_cloth = load_image("/workspace1/pdawson/tryon-scraping/dataset/test/cloth/d80d5841899a4eb8850a8a7a6df901de32cf51c5.jpg")

size = (768, 1152)
test_cloth = np.array(load_image("/workspace1/pdawson/tryon-finetune/wolfram/cargo.png"))
target_ratio = size[1] / size[0]
current_ratio = test_cloth.shape[0] / test_cloth.shape[1]

if current_ratio < target_ratio:
    # vertical padding
    pad_height = int(test_cloth.shape[1] * target_ratio - test_cloth.shape[0])
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    test_cloth = np.pad(test_cloth, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
else:
    # horizontal padding
    pad_width = int(test_cloth.shape[0] / target_ratio - test_cloth.shape[1])
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    test_cloth = np.pad(test_cloth, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=255)

test_cloth = Image.fromarray(test_cloth).resize(size)
test_cloth = np.array(test_cloth)
model_input = np.column_stack([test_cloth, np.ones_like(test_cloth)*255])
model_input = Image.fromarray(model_input)
model_input.save("model_input.png")

cloth_prompt = "Cargo pants"
model_prompt = "Man in front of a clean white background"
prompt_detailed = create_prompt(model_prompt, cloth_prompt)
#prompt_detailed = prompt_generic
print(prompt_detailed)



result = pipe(
        prompt=prompt_generic,
        prompt_2=prompt_detailed,
        height=size[1],
        width=size[0]*2,
        image=model_input,
        num_inference_steps=40,
        guidance_scale=4,
        strength=1.0,
        joint_attention_kwargs={"scale": 0.8}
).images

result[0].save("/workspace1/pdawson/tryon-finetune/cargo.png")