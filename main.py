from model import VGG16
from train import train_loop
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image_path', type = str, help = 'Image to which we will approximate our generate image', default = 'content_image.jpg')
    parser.add_argument('--style_image_path', type = str, help = 'Image which style will be casting to our generate image', default = 'style_image.jpg')
    parser.add_argument('--steps', type = int, help = 'Number of steps for training', default = 300)
    parser.add_argument('--lr', type = float, help = 'Learning rate', default = 0.01)
    parser.add_argument('--Height', type = int, help = 'Height of generated image', default = 224)
    parser.add_argument('--Width', type = int, help = 'Width of generated image', default = 224)
    parser.add_argument('--model', type = str, help = 'Model name', default = 'VGG16')
    parser.add_argument('--optimizer', type = str, help = 'Optimizer name', default = 'LBFGS')
    parser.add_argument('--alpha', type = float, help = 'Weight of content loss', default = 1.0)
    parser.add_argument('--beta', type = float, help = 'Weight of style loss', default = 0.01)
    parser.add_argument('--variation', type = float, help = 'Weight of variation loss', default = 1e-4)
    args = parser.parse_args()
    
    # model = VGG16()
    # print(model.pretrained_weights)
    
    print(args)
    train_loop(args)
    
    