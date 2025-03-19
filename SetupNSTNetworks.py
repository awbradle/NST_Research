import torch

from torchvision.models import googlenet, GoogLeNet_Weights
def setup_googlenet():
    cnn = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [10]
    style_layers = [3,7,10]
    if hasattr(cnn, "features"):
        cnn.features.eval()
    else:
        cnn.eval()
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers]