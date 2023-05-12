import torch
from sae_frontend_new import SAE
from backend_regressor import make_layers, Back_Regr_Net, build_back_regr
import torch.nn as nn
from utils import _initialize_weights, params_count_lite
import torchvision.transforms as T

class ScaleNet_Multi(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.frontend = SAE()
        
        self.back1 = Back_Regr_Net(d_rate=2, multi_sae=True).backend
        self.back2 = Back_Regr_Net(d_rate=3, multi_sae=True).backend
        self.back3 = Back_Regr_Net(d_rate=4, multi_sae=True).backend

        self.regressor = nn.Sequential(
            nn.Linear(4096*3, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        lat, _ = self.frontend(x)
        b1 = self.back1(lat)
        b2 = self.back2(lat)
        b3 = self.back3(lat)

        total_b = torch.cat([b1, b2, b3], dim=1)

        res = self.regressor(total_b)
        return res


def build_scalenet_multi(pretrained=False, pretrained_sae=False, freeze_sae=False):
    model = ScaleNet_Multi()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pretrained:
        checkpoint = torch.load('./weights/model_multi_gray.pth', map_location='cpu')
        print(checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        if pretrained_sae:
            sae_chck = torch.load('./sae_weights/ae123_best.pth')
            model.frontend.load_state_dict(sae_chck['model_state_dict'], strict=False)
    
    for param in model.frontend.parameters():
        param.requires_grad = not freeze_sae

    return model


transform = torch.nn.Sequential(
    T.ToTensor(),
    # T.Grayscale(num_output_channels=3),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    # T.CenterCrop(size=(1035, 1035))
)
model = build_scalenet_multi()

total_model = torch.nn.Sequential(transform, model)
print(total_model)


# modules = [model.frontend, model.back1, model.back2, model.back3]
# for module in modules:
#     for p in module.parameters():
#         p.requires_grad = False


# params_count_lite(model)


        
    

    
    

