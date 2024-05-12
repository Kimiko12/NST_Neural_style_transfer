from load_image import load_image, HEIGHT, save_image
from PIL import Image
import torch
from tqdm import tqdm
from model import VGG16
from torch.optim import LBFGS
from utils import customize_model
import torchvision.transforms as transforms
import cv2

transform = transforms.Compose([
    transforms.Resize(HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

def Gram_matrix(feature_map):
    batch_size, channel, height, width = feature_map.shape
    # change shape of input tensors from 4 - dimensional to 2 - dimensional vector
    features = feature_map.view(batch_size * channel, height * width)
    Gram = torch.mm(features, features.t())
    
    return Gram.div(batch_size * channel * height * width)


def variation_loss(generated_feature_map):
    x_variation = generated_feature_map[:,:,1:,:] - generated_feature_map[:,:,:-1,:]
    y_variation = generated_feature_map[:,:,:,1:] - generated_feature_map[:,:,:,:-1]
    return torch.mean(x_variation**2) + torch.mean(y_variation**2)
    

# def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_image = load_image(args.content_image_path, device = device, transform = transform, shape = (args.Height, args.Width))
    style_image = load_image(args.style_image_path, device = device, transform = transform, shape = (args.Height, args.Width))
    generated_image = torch.rand_like(content_image, device=device, requires_grad=True)
    
    
    loop = tqdm(range(args.steps), total=args.steps, desc='Training process')
    model = VGG16().to(device)
    for i in loop: 
        generated_feature_maps = model(generated_image)
        content_feature_maps = model(content_image)
        style_feature_maps = model(style_image)
        
        style_loss = content_loss = 0
        
        for generated_feature_map, content_feature_map, style_feature_map in zip(generated_feature_maps, content_feature_maps, style_feature_maps):
            
            batch_size, channel, height, width = generated_feature_map.shape
            # calculate content loss
            content_loss += torch.mean(generated_feature_map - content_feature_map)**2
            
            # calculate style loss
            
            Gram_matrix_generated = Gram_matrix(generated_feature_map)
            Gram_matrix_style = Gram_matrix(style_feature_map)
            
            style_loss += torch.mean((Gram_matrix_style - Gram_matrix_generated)**2 / (channel * height * width))
            
            total_loss = args.alpha * content_loss + args.beta * style_loss + args.variation * variation_loss(generated_feature_map)
            
        if args.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS((generated_feature_map), lr = args.lr)
        else:
            optimizer = torch.optim.Adam((generated_feature_map), lr = args.lr)
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loop.set_description(f'Step {i+1}/{args.steps}')
        loop.refresh()
        
        if args.steps % 100 == 0:
            print(f'Total loss: {total_loss.item()}')
            save_image(generated_image, './output_images', f'{i}.jpg')
                
def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_image = load_image(args.content_image_path, device=device, transform=transform, shape=(args.Height, args.Width))
    style_image = load_image(args.style_image_path, device=device, transform=transform, shape=(args.Height, args.Width))
    generated_image = torch.rand_like(content_image, device=device, requires_grad=True)
    
    model = VGG16().to(device)
    optimizer = torch.optim.LBFGS([generated_image.requires_grad_()], lr=args.lr) if args.optimizer == 'LBFGS' else torch.optim.Adam([generated_image.requires_grad_()], lr=args.lr)

    loop = tqdm(range(args.steps), total=args.steps, desc='Training process')
    for i in loop:
        def closure():
            optimizer.zero_grad()
            
            generated_feature_maps = model(generated_image)
            content_feature_maps = model(content_image)
            style_feature_maps = model(style_image)

            style_loss = content_loss = 0

            for (_, generated_feature_map), (_, content_feature_map), (_, style_feature_map) in zip(generated_feature_maps, content_feature_maps, style_feature_maps):

                # calculate content loss
                content_loss += torch.mean((generated_feature_map - content_feature_map) ** 2)

                # calculate style loss
                G = Gram_matrix(generated_feature_map)
                A = Gram_matrix(style_feature_map)
                style_loss += torch.mean((G - A) ** 2) / (generated_feature_map.shape[1] * generated_feature_map.shape[2] * generated_feature_map.shape[3])

            total_loss = args.alpha * content_loss + args.beta * style_loss + args.variation * variation_loss(generated_image)
            total_loss.backward()

            loop.set_description(f"Step {i+1}/{args.steps}, Total loss: {total_loss.item()}")
            return total_loss

        optimizer.step(closure)

        if i % 100 == 0:
            save_image(generated_image.detach(), './output_images', f'{i}.jpg')
           
            
            
            
            