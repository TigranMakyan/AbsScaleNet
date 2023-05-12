from tkinter import TRUE
import torch
import torch.nn as nn

from utils import _initialize_weights


def make_layers(cfg, in_channels = 3,batch_norm=True, dilation = True, d=2):
    if dilation:
        d_rate = d
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.Flatten()]
    return nn.Sequential(*layers)

class Back_Regr_Net(nn.Module):
    def __init__(self, d_rate=2, multi_sae=False):
        super(Back_Regr_Net, self).__init__()
        if multi_sae:
            self.backend_feat = [512, 512, 256, 128, 64]
        else:
            self.backend_feat  = [512, 512, 512, 256, 256, 128 ,64, 64, 32, 16]
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True, d=d_rate)
        self.regressor = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
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

