import torch
import torch.nn as nn

from vgg_frontend import build_vgg_as_frontend, vgg16_bn
from backend_regressor import build_back_regr, Back_Regr_Net
from utils import _initialize_weights

class ScaleNet_vgg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_model= vgg16_bn(pretrained=False)
        self.frontend = nn.Sequential(*list(vgg_model.features.children()))
        # self.frontend.add_module('flatten',nn.Flatten())
        self.back_regr = Back_Regr_Net()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.frontend(x)
        x = self.back_regr(x)
        return x


def load_model(checkpoint_path, model, optimizer, criterion):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def build_scalenet_vgg(pretrained, pretrained_vgg=False, pretrained_back_regr=False, freeze_vgg=False, freeze_back_regr=False, leaky=False):
    model = ScaleNet_vgg()

    if pretrained:
        print('Loading pretrained weights for model')
        # model.load_state_dict(torch.load('./weights/model_vgg.pth'), strict=False)  
        checkpoint = torch.load('./weights/model_vgg_self_norm_sgd.pth')
        model.load_state_dict(checkpoint['model_state_dict'])   
    else:
        _initialize_weights(model)
        if pretrained_vgg:
            model.frontend.load_state_dict(torch.load('./weights/only_vgg.pth'), strict=False)
        
        if pretrained_back_regr:
            model.back_regr.load_state_dict(torch.load('./weights/back_regr.pth'), strict=False) 

    if freeze_vgg:
        for params in model.frontend.parameters():
            params.requires_grad = False
    elif not freeze_vgg:
        for params in model.frontend.parameters():
            params.requires_grad = True  

    if freeze_back_regr:
        for params in model.back_regr.parameters():
            params.requires_grad = False
    else:
        for params in model.back_regr.parameters():
            params.requires_grad = True

    if leaky:
        l = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]
        for act in l:
            model.frontend[act] = torch.nn.LeakyReLU(inplace=True)
    return model


# model = build_scalenet_vgg(False, False, False, True, False)
# print(model)# print(a.shape)
# res = model.frontend(a)
# print(res.shape)
# total_params = sum(p.numel() for p in model.parameters())
# tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(total_params, tr_params)
        
    

        
