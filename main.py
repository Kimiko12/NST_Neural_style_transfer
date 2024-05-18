from __future__ import division
import argparse
import torch
from train2 import train_loop, train_loop_for_video
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_size', type=int, default=900)
    parser.add_argument('--total_step', type=int, default=1000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--video_path', type=str, default='sample.mp4')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_path', type=str, default='output.mp4')
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=1e1)
    parser.add_argument('--content', type = str, help = 'Image to which we will approximate our generate image', default = 'content_image.jpg')
    parser.add_argument('--style', type = str, help = 'Image which style will be casting to our generate image', default = 'style_image.jpg')
    parser.add_argument('--model', type = str, help = 'Model name', default = 'VGG19')
    parser.add_argument('--optimizer', type = str, help = 'Optimizer name', default = 'LBFGS')
    parser.add_argument('--alpha', type = float, help = 'Weight of content loss', default = 1e5)
    parser.add_argument('--beta', type = float, help = 'Weight of style loss', default = 3e4)
    parser.add_argument('--variation', type = float, help = 'Weight of variation loss', default = 1e0)
    parser.add_argument('--variant_of_start_generation', type = str, help = 'Start generation with content or Gaussian noise', default = 'Gaussian_noise')
    config = parser.parse_args()
    print(config)
    # train_loop(config)
    
    config.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
    ])
    
    
    train_loop_for_video(config)