from PIL import Image
import torch
from tqdm import tqdm
from load_image import load_image, save_image
from model import VGG19
from torch.optim import LBFGS
import torchvision.transforms as transforms
import cv2
import torchvision
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])

def Gram_matrix(feature_map):
    #_, channel, height, width = feature_shape
    _, channel, height, width = feature_map.size()
    # change shape of input tensors from 4 - dimensional to 2 - dimensional vector
    features = feature_map.view(channel, height * width)
    Gram = torch.mm(features, features.t())
    
    # return Gram.div(channel * height * width)
    return Gram

def total_variation(optimizing_img):
    return torch.sum(torch.abs(optimizing_img[:, :, :, :-1] - optimizing_img[:, :, :, 1:])) + \
           torch.sum(torch.abs(optimizing_img[:, :, :-1, :] - optimizing_img[:, :, 1:, :]))

def train_loop(config):
    content_image = load_image(config.content, transform = transform, max_size=config.max_size)
    style_image = load_image(config.style, transform = transform, shape=[content_image.size(2), content_image.size(3)])
    if config.variant_of_start_generation == 'Gaussian_noise':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size = content_image.shape).astype(np.float32)
        init_image = torch.from_numpy(gaussian_noise_img).float().to(device) 
    else: 
        # optimizing_img = content_image.clone().requires_grad_(True)
        init_image = content_image
        
    optimizing_img = Variable(init_image, requires_grad=True)
    model = VGG19().to(device).eval()
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam([optimizing_img], lr=config.lr, betas=[0.5, 0.999])
        loop = tqdm(range(config.total_step), total=config.total_step, desc='Training process')
        for i in loop:
            optimizer.zero_grad()
            optimizing_feature_maps = model(optimizing_img)
            content_feature_maps = model(content_image)
            style_feature_maps = model(style_image)
            
            style_loss = content_loss = 0

        for optimizing_feature_map, content_feature_map, style_feature_map in zip(optimizing_feature_maps, content_feature_maps, style_feature_maps):
            content_loss += torch.mean((optimizing_feature_map - content_feature_map)**2)

            _, chanel, height, width = optimizing_feature_map.size()
            
            G = Gram_matrix(optimizing_feature_map)
            A = Gram_matrix(style_feature_map)
            
            style_loss += torch.mean((G - A)**2) / (chanel * height * width) 
        
        total_variation = total_variation(optimizing_img)
        
        loss = config.alpha * content_loss + config.beta * style_loss + config.variation * total_variation
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % config.log_step == 0:
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                   .format(i+1, config.total_step, content_loss.item(), style_loss.item()))

        if (i+1) % config.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = optimizing_img.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(i+1))