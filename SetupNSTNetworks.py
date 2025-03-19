import torch

from torchvision.models import densenet161, DenseNet161_Weights, googlenet, GoogLeNet_Weights, resnet152, ResNet152_Weights, squeezenet1_0, SqueezeNet1_0_Weights, vgg19, VGG19_Weights
def setup_DenseNet():
    cnn = densenet161(weights=DenseNet161_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [5]
    style_layers = [5,7,9]
    style_weight = 8000000000000
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers, style_weight]

def setup_GoogleNet():
    cnn = googlenet(weights=GoogLeNet_Weights.DEFAULT).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [10]
    style_layers = [3,7,10]
    style_weight = 10000000
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers, style_weight]

squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).features.eval()
def setup_SqueezeNet():
    cnn = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [6]
    style_layers = [4,5,6,8,9,10,11,13]
    style_weight = 1000000000
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers, style_weight]

def setup_ResNet():
    cnn = resnet152(weights=ResNet152_Weights.DEFAULT).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [5]
    style_layers = [5,6,7]
    style_weight = 100000000
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers, style_weight]

def setup_VGG():
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    content_layers = [8]
    style_layers = [3,6,8,11]
    style_weight = 10000000
    return [cnn, cnn_normalization_mean, cnn_normalization_std, content_layers, style_layers, style_weight]
