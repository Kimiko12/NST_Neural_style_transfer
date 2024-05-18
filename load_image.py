import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def load_image(image_path, transform=None, max_size=None, shape=None):
    if os.path.exists(image_path):
        if image_path.endswith('png') or image_path.endswith('jpg'):
            image = Image.open(image_path).convert('RGB') 
        else:
            raise ValueError('{} is not a valid image file'.format(image_path))
    else:
        raise ValueError('{} does not exist'.format(image_path))
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.Resampling.LANCZOS)
    if shape:
        image = image.resize(shape, Image.Resampling.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)

def save_image(image_tensor, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_np = image_tensor.cpu().numpy()
    
    if image_np.ndim == 4:
        image_np = image_np[0]

    if image_np.shape[0] == 3:  
        image_np = np.transpose(image_np, (1, 2, 0))
    
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255.0
    image_np = image_np.astype(np.uint8)

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"Image saved to {save_path}")