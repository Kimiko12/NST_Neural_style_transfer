import torch
import torch.nn as nn
from  torchvision import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class VGG16(nn.Module):
    def __init__(self, requires_grad = False):
        super(VGG16, self).__init__()
                
        self.pretrained_weights = models.vgg16(pretrained=True, progress=True).features[:29]
        self.pretrained_weights.to(device)
        # self.hidden_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        
        # self.hidden_layers['conv1'] = self.pretrained_weights[0]
        # self.hidden_layers['conv2'] = self.pretrained_weights[5]
        # self.hidden_layers['conv3'] = self.pretrained_weights[10]
        # self.hidden_layers['conv4'] = self.pretrained_weights[17]
        # self.hidden_layers['conv5'] = self.pretrained_weights[24]
        
        self.hidden_layers = {
            'conv1': self.pretrained_weights[0],
            'conv2': self.pretrained_weights[5],
            'conv3': self.pretrained_weights[10],
            'conv4': self.pretrained_weights[17],
            'conv5': self.pretrained_weights[24]
        }
        
        if requires_grad is False:
            for layer in self.hidden_layers.values():
                for param in layer.parameters():
                    param.requires_grad = False
        
        for layer in self.hidden_layers.values():
            layer.to(device)
        
        self.content_feature_map_index = 0
        self.style_feature_map_indices = list(range(len(self.hidden_layers)))
            
    
    def forward(self, x):
        feature_maps = []
        for name, layer in self.hidden_layers.items():
            x = layer(x)
            if name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                feature_maps.append((name, x))
        return feature_maps
        