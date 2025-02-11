# coding=utf-8
from typing import Any, Optional, Tuple, Literal

import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.utils.data import DataLoader
import json

debug_mode=False

def tensor_to_image(tensor, image_path):
    """
    Convert a torch tensor to an image file.

    Args:
    - tensor (torch.Tensor): the input tensor. Shape (C, H, W).
    - image_path (str): path where the image should be saved.

    Returns:
    - None
    """
    if debug_mode: 
        # Check the tensor dimensions. If it's a batch, take the first image
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Check for possible normalization and bring the tensor to 0-1 range if necessary
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Convert tensor to PIL Image
        to_pil = ToPILImage()
        img = to_pil(tensor)

        # Save the PIL Image
        dir_path = os.path.dirname(image_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img.save(image_path)

def create_prompt(model_prompt, cloth_prompt):

    prompt = f"The pair of images highlights a garment and its styling on a model; "
    prompt += f"[IMAGE1] {cloth_prompt};"
    prompt += f"[IMAGE2] {model_prompt};"

    return prompt

def create_prompt_generic(model_prompt, cloth_prompt):

    prompt = f"The pair of images highlights a garment and its styling on a model; "
    prompt += f"[IMAGE1] Detailed photo of a garment;"
    prompt += f"[IMAGE2] {model_prompt}. The person is wearing the garment;"

    return prompt


class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        data_list: Optional[str] = None
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        # This code defines a transformation pipeline for image processing
        self.transform = transforms.Compose(
            [
                # Convert the input image to a PyTorch tensor
                transforms.ToTensor(),
                # Normalize the tensor values to a range of [-1, 1]
                # The first [0.5] is the mean, and the second [0.5] is the standard deviation
                # This normalization is applied to each color channel
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cloth_transform = transforms.Compose(
            [
            transforms.ToTensor()
            ]
        )

        self.toTensor = transforms.ToTensor()

        self.order = order
        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []
        prompts = []

        filename = os.path.join(dataroot_path, data_list)
        
        with open(filename, "r") as f:
            for line in f.readlines():
                if order == "paired":
                    c_name, im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    c_name, im_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)                
                        
        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.prompts = prompts

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        # Cloth image
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).resize((self.width,self.height))
        cloth_pure = self.transform(cloth)
        cloth_pure_01 = self.cloth_transform(cloth)
        
        # Model image
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)
        
        # Prompt
        cloth_prompt = open(os.path.join(self.dataroot, self.phase, "cloth", c_name + ".txt")).read()
        image_prompt = open(os.path.join(self.dataroot, self.phase, "image", im_name + ".txt")).read()
        prompt = create_prompt(image_prompt, cloth_prompt)
        #prompt = create_prompt_generic(image_prompt, cloth_prompt)

        # Inpaint mask
        garment_mask = torch.zeros_like(cloth_pure)
        full_mask = torch.cat([garment_mask, 1.0-garment_mask], dim=2) # Full screen "inpaint"

        # Loss mask
        loss_mask = torch.ones(1, self.height, self.width*2)
        loss_mask[:, :, :self.width] = 0

        if (os.path.exists(os.path.join(self.dataroot, self.phase, "mask", im_name))):
            empty_mask = torch.zeros(1, self.height, self.width)
            mask = Image.open(os.path.join(self.dataroot, self.phase, "mask", im_name)).resize((self.width,self.height)).convert("L")
            mask = self.toTensor(mask)
            mask = (mask > 0.5).float()
            mask = torch.cat([empty_mask, mask], dim=2)
            loss_mask = mask

        loss_mask = torch.clamp(loss_mask, 0.6, 1)
        
        # Save all
        result = {}
        result["inpaint_mask"] = full_mask
        result["c_name"] = c_name
        result["im_name"] = im_name
        #result["cloth_pure"] = cloth_pure
        result["prompt"] = prompt
        result["loss_mask"] = loss_mask
        result["cloth_pure_01"] = cloth_pure_01
        
        # Concatenate image and garment along width dimension
        inpaint_image = torch.cat([cloth_pure, torch.zeros_like(cloth_pure)], dim=2)  # dim=2 is width dimension
        result["im_mask"] = inpaint_image
        
        GT_image = torch.cat([cloth_pure, image], dim=2)  # dim=2 is width dimension
        result["image"] = GT_image
        
        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)


if __name__ == "__main__":
    dataset = VitonHDDataset("/workspace1/pdawson/tryon-scraping/dataset2", "train", 
                             "unpaired", (768,512), "train_pairs.txt")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for data in loader:
        pass
