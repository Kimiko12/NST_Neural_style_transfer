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
from load_video import VideoDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
])

# def Gram_matrix(feature_map):
#     if feature_map.ndim == 4:
#         _, channel, height, width = feature_map.size()
#     elif feature_map.ndim == 3:
#         channel, height, width = feature_map.size()
#     features = feature_map.view(channel, height * width)
#     Gram = torch.mm(features, features.t())
#     return Gram

def Gram_matrix(feature_map):
    if isinstance(feature_map, list):
        return [Gram_matrix(f) for f in feature_map]
    _, d, h, w = feature_map.size()
    feature_map = feature_map.view(d, h * w)
    gram = torch.mm(feature_map, feature_map.t())
    return gram


def total_variation_loss(optimizing_img):
    return torch.sum(torch.abs(optimizing_img[:, :, :, :-1] - optimizing_img[:, :, :, 1:])) + \
           torch.sum(torch.abs(optimizing_img[:, :, :-1, :] - optimizing_img[:, :, 1:, :]))
           
num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

def train_loop(config):
    content_image = load_image(config.content, transform=transform, max_size=config.max_size)
    style_image = load_image(config.style, transform=transform, shape=[content_image.size(2), content_image.size(3)])
    
    if config.variant_of_start_generation == 'Gaussian_noise':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_image.shape).astype(np.float32)
        init_image = torch.from_numpy(gaussian_noise_img).float().to(device)
    else: 
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
            
            style_loss = 0
            content_loss = 0

            for optimizing_feature_map, content_feature_map, style_feature_map in zip(optimizing_feature_maps, content_feature_maps, style_feature_maps):
                content_loss += torch.mean((optimizing_feature_map - content_feature_map) ** 2)
                _, channel, height, width = optimizing_feature_map.size()
                
                G = Gram_matrix(optimizing_feature_map)
                A = Gram_matrix(style_feature_map)
                
                style_loss += torch.mean((G - A) ** 2) / (channel * height * width)

            tv_loss = total_variation_loss(optimizing_img)
            
            loss = config.alpha * content_loss + config.beta * style_loss + config.variation * tv_loss
            
            loss.backward()
            optimizer.step()

            if (i + 1) % config.log_step == 0:
                print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                      .format(i + 1, config.total_step, content_loss.item(), style_loss.item()))

            if (i + 1) % config.sample_step == 0:
                denorm = transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), 
                                              (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225))
                img = optimizing_img.clone().squeeze()
                img = denorm(img).clamp_(0, 1)
                torchvision.utils.save_image(img, 'output-{}.png'.format(i + 1))
    if config.optimizer == 'LBFGS':
        optimizer = LBFGS([optimizing_img], max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0
        
        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            optimizing_feature_maps = model(optimizing_img)
            content_feature_maps = model(content_image)
            style_feature_maps = model(style_image)

            style_loss = 0
            content_loss = 0

            for optimizing_feature_map, content_feature_map, style_feature_map in zip(optimizing_feature_maps, content_feature_maps, style_feature_maps):
                content_loss += torch.mean((optimizing_feature_map - content_feature_map) ** 2)
                _, channel, height, width = optimizing_feature_map.size()

                G = Gram_matrix(optimizing_feature_map)
                A = Gram_matrix(style_feature_map)

                style_loss += torch.mean((G - A) ** 2) / (channel * height * width)

            tv_loss = total_variation_loss(optimizing_img)

            loss = config.alpha * content_loss + config.beta * style_loss + config.variation * tv_loss
            loss.backward()

            print(f'Step [{cnt + 1}/{config.total_step}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}')
            
            if (cnt + 1) % config.sample_step == 0:
                denorm = transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), 
                                            (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225))
                img = optimizing_img.clone().squeeze()
                img = denorm(img).clamp_(0, 1)
                torchvision.utils.save_image(img, f'output-{cnt + 1}.png')
            
            cnt += 1
            return loss

        for i in tqdm(range(config.total_step), total=config.total_step, desc='Training process'):
            optimizer.step(closure)
            
            
# ------------------------------------------------------------------------------------------------
     
def save_video(frames, output_path, fps):
    height, width = frames[0].shape[0:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()



def Gram_matrix_for_video(feature_map):
    if isinstance(feature_map, list):
        return [Gram_matrix(f) for f in feature_map]
    d, h, w = feature_map.size()
    feature_map = feature_map.view(d, h * w)
    gram = torch.mm(feature_map, feature_map.t())
    return gram

def train_loop_for_video(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading dataset...")
    dataset = VideoDataset(video_path=config.video_path, transform=config.transform, max_size=config.max_size)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    for step, batch in enumerate(dataloader):
        print(batch.shape)
    print("Loading style image...")
    style_image = load_image(config.style, transform=config.transform, max_size=config.max_size).to(device)
    
    model = VGG19().to(device).eval()
    
    # Transfer style_image to necessary shape (3,3,516,900) 
    style_image = style_image.clone().detach().requires_grad_()

    if config.optimizer == 'LBFGS':
        num_of_iterations = {'lbfgs': config.total_step}
        processed_frames = []
        for epoch in range(config.total_step):
            for step, batch in enumerate(dataloader):
                if config.variant_of_start_generation == 'Gaussian_noise':
                    # print(f'Batch[0] shape: {batch[0].shape}')
                    # print(f'Batch shape: {batch.shape}')
                    gaussian_noise_img = np.random.normal(loc=0, scale=90., size=batch[0].shape).astype(np.float32)
                    gaussian_noise_img_batch = np.stack([gaussian_noise_img for _ in range(config.batch_size)], axis=0)
                    gaussian_noise_img_batch = torch.from_numpy(gaussian_noise_img_batch).float().to(device).requires_grad_()
                    # print(f'Size gaussian_noise_img_batch: {gaussian_noise_img_batch.shape}')
                    style_image_resized = F.interpolate(style_image, size=(batch.shape[2], batch.shape[3]), mode='bilinear', align_corners=False)
                    style_img_batch = style_image_resized.expand(batch.size(0), -1, -1, -1)
                    # print(f'Style_img_batch shape: {style_img_batch.shape}')
                for optimizing_image in gaussian_noise_img_batch:
                    optimizing_image = optimizing_image.clone().detach().requires_grad_()
                    # print(optimizing_image.shape)

                optimizer = LBFGS([gaussian_noise_img_batch], max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')

                def closure():
                    optimizer.zero_grad()
                    
                    batch_style_loss = 0
                    batch_content_loss = 0
                    
                    content_image_batch = batch.to(device)
                    
                    assert(style_img_batch.shape == content_image_batch.shape == gaussian_noise_img_batch.shape, \
                    f"Shapes do not match: {content_image_batch.shape}, {style_img_batch.shape}, {gaussian_noise_img_batch.shape}, {batch.shape}")

                    optimizing_features_batch = (model(gaussian_noise_img) for gaussian_noise_img in gaussian_noise_img_batch)
                    content_features_batch = (model(content_image_batch) for content_image_batch in content_image_batch)
                    style_features_batch = (model(style_img_batch) for style_img_batch in style_img_batch)

                    # print(gaussian_noise_img_batch.shape)
                    # print(content_image_batch.shape)
                    # print(style_img_batch.shape)
                    
                    # for image in optimizing_features_batch:
                    #     for optimizing_feature in image:
                    #         print(optimizing_feature.shape)
                    
                    content_loss = 0
                    style_loss = 0
                    counter = 0
                    for optimizing_features, content_features, style_features in zip(optimizing_features_batch, content_features_batch, style_features_batch):
                        for optimizing_feature, content_feature, style_feature in zip(optimizing_features, content_features, style_features):
                            
                            content_loss = torch.mean((optimizing_feature - content_feature) ** 2)
                            channels, height, width = optimizing_feature.size()
                            
                            G = Gram_matrix_for_video(optimizing_feature)
                            A = Gram_matrix_for_video(style_feature)
                            
                            style_loss += torch.mean((G - A) ** 2/ (height * width * channels))
                            
                            batch_content_loss += content_loss
                            batch_style_loss += style_loss
                            # tv_loss = total_variation_loss(optimizing_image)
                            counter += 1
                            
                    # print(counter)
                    # total_loss = config.alpha * (batch_content_loss / config.batch_size) + config.beta * (batch_style_loss / config.batch_size) + config.tv_weight * tv_loss / config.batch_size
                    total_loss = config.alpha * (batch_content_loss / config.batch_size) + config.beta * (batch_style_loss / config.batch_size)
                    total_loss.backward()
                    print(f'Total loss per batch: {total_loss.item()}')
                    return total_loss
                optimizer.step(closure)
                    
                    
            if (epoch + 1) % 1 == 0:
                print(f"Step {step + 1}/{len(dataloader)}, Loss: {closure().item()}")
                
                
            torch.cuda.empty_cache()


