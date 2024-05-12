import torch 
from model import VGG16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def customize_model(model, device):
    if model == 'VGG16':
        model = model(requires_grad=False, show_progress=True)
        model.to(device)
    else:
        raise ValueError(f'{model} not found !!!')
        
    content_feature_map_index = model.content_feature_map_index
    style_feature_map_indices = model.style_feature_map_indices
    layer_name = model.hidden_layers