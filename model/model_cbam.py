import torch
import torch.nn as nn

from resnet_cbam import ResNetCBAM50, ResNetCBAM18, ResNetCBAM34
from backend_regressor import make_layers, Back_Regr_Net
from utils import _initialize_weights

class BRNet(nn.Module):
    def __init__(self):
        super(BRNet, self).__init__()
        self.backend_feat  = [2048, 1024, 512, 256, 256, 128 ,64, 64, 32, 16]
        self.backend = make_layers(self.backend_feat,in_channels = 2048,dilation = True, batch_norm=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(16, 1)
        )
        
    
    def forward(self,x):
        x = x.to(torch.float32)
        x = self.backend(x)
        x = self.regressor(x)
        return x


def build_back_regr(pretrained=False, freeze=False):
    model = Back_Regr_Net()
    if pretrained:
        print('[INFO]: Loading pretrained weights for backend and regressor blocks')
        model.load_state_dict(torch.load('./weights/back_regr.pth'))
    else:
        print('[INFO]: NOT Loading pretrained weights for backend and regressor blocks')
        _initialize_weights(model)
    if freeze:
        print('[INFO]: FREEZE backend and regressor')
        for params in model.parameters():
            params.requires_grad = False
    else:
        print('[INFO]: BackPropogation via backend and regressor')
        for params in model.parameters():
            params.requires_grad = True

    return model


class ScaleNet_CBAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontent = ResNetCBAM34()
        self.back_regr = build_back_regr()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.frontent(x)
        x = self.back_regr(x)
        return x


def build_cbam_model(pretrained=False, freeze_resnet=False, freeze_back_regr=False):
    model = ScaleNet_CBAM()
    if pretrained:
        print('[INFO]: Loading pretrained weights for the entire CBAM network')
        checkpoint = torch.load('./weights/model_cbam.pth')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        _initialize_weights(model)
    if freeze_resnet:
        for p in model.frontent.parameters():
            p.requires_grad = False
    if freeze_back_regr:
        for p in model.back_regr.parameters():
            p.requires_grad = False
    return model


# model = build_cbam_model(False, False, False)
# image = torch.rand(1, 3, 256, 256)
# res = model(image)
# print(res.shape)

# print(sum(p.numel() for p in model.parameters()))
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))