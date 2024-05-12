import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import cv2

HEIGHT = 224 
WIDTH = 224

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, device, transform = None, shape = (HEIGHT, WIDTH)):
    if os.path.exists(image_path):
        if image_path.endswith('jpg'):
            image = Image.open(image_path).convert('RGB')
        elif image_path.endswith('png'):
            # image_path = image_path.replace('png', 'jpg')
            image = Image.open(image_path).convert('RGB')
    else:
        raise ValueError(f'{image_path} not found !!!')
    
    if image.size != shape:
        image = image.resize(shape, Image.ANTIALIAS)
    
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    image = Image.fromarray((image * 255).astype(np.uint8))
    if transform:
        image = transform(image).unsqueeze(0).to(device)
        
    return image


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


# def save_image(image, path_to_save = './output_images', filename = 'result.jpg'):
    # if not os.path.exists(path_to_save):
    #     os.mkdir(path_to_save)
    
    # if torch.is_tensor(image):
    #     image = image.clone().detach().cpu().numpy()
    #     image = image.transpose(1,2,0)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    # full_path = os.path.join(path_to_save, filename)
    # cv2.imwrite(full_path, image)