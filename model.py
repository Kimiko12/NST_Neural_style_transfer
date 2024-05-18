import torch
import torch.nn as nn
from  torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG19, self).__init__()
        self.names = ['0', '5', '10', '19', '28'] 
        self.pretrained_model = models.vgg19(pretrained=True, progress=True).features
        self.hidden_layers = {
            'conv1': self.pretrained_model[0],
            'conv2': self.pretrained_model[5],
            'conv3': self.pretrained_model[10],
            'conv4': self.pretrained_model[19],
            'conv5': self.pretrained_model[28]
        }
        
        if requires_grad is False:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
                
        for layer in self.hidden_layers.values():
            layer.to(device)
            
        self.content_feature_map_index = 0
        self.style_feature_map_indices = list(range(len(self.hidden_layers)))
        
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.pretrained_model._modules.items():
            x = layer(x)
            if name in self.names:
                features.append(x)
        return features