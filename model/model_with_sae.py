import torch
import torch.nn as nn

from backend_regressor import build_back_regr, Back_Regr_Net
from sae_frontend import build_sae_as_frontend, StackedAutoEncoder


class ScaleNet_sae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontent = StackedAutoEncoder()
        self.back_regr = Back_Regr_Net()

    def forward(self, x):
        x = x.to(torch.float32)
        x, rec = self.frontent(x)
        x = self.back_regr(x)
        return x

def build_scalenet_sae(pretrained, pretrained_sae=False, pretrained_back_regr=False, freeze_sae=False, freeze_back_regr=False):
    model = ScaleNet_sae()
    if pretrained:
        print('[INFO]: Loading pretrained weights for the entire network')
        model.load_state_dict(torch.load('./weights/model_vgg.pth'))
        return model
    else:
        print('[INFO]: NOT loading pretrained weights for the entire network')
        model.frontent = build_sae_as_frontend(pretrained_sae, freeze=freeze_sae)
        model.back_regr = build_back_regr(pretrained_back_regr, freeze=freeze_back_regr)
        
    return model

