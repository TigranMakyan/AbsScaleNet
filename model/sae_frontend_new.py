import torch
import torch.nn as nn
import torch.optim as optim

encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 32, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )
decoder1 = nn.Sequential(
    nn.ConvTranspose2d(32, 16, 3, 1, 1), 
    nn.BatchNorm2d(16),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(16, 8, 5, 2, 1), 
    nn.BatchNorm2d(8),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(8, 3, 2, 2, 1),
    nn.BatchNorm2d(3),
    nn.LeakyReLU(0.1, True)
)

encoder2 = nn.Sequential(
    nn.Conv2d(32, 128, 3, 1, 1),
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.1, True),
    nn.Conv2d(128, 256, 3, 1, 1),
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.1, True)
)
decoder2 = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 3, 1, 1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(128, 64, 5, 2, 1), 
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(64, 32, 2, 2, 1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1, True),
)
encoder3 = nn.Sequential(
    nn.Conv2d(256, 512, 3, 1, 1),
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.1, True),
    nn.Conv2d(512, 512, 3, 1, 1), 
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.1, True)
)
decoder3 = nn.Sequential(
    nn.ConvTranspose2d(512, 256, 3, 1, 1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(256, 256, 4, 2, 1), 
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.1, True),
)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        lat = self.encoder(x)
        out = self.decoder(lat)
        return lat, out


class SAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ae1 = AutoEncoder(encoder=encoder1, decoder=decoder1)
        self.ae2 = AutoEncoder(encoder=encoder2, decoder=decoder2)
        self.ae3 = AutoEncoder(encoder=encoder3, decoder=decoder3)

    def forward(self, x):
        enc1 = self.ae1.encoder(x)
        enc2 = self.ae2.encoder(enc1)
        lat = self.ae3.encoder(enc2)
        dec3 = self.ae3.decoder(lat)
        dec2 = self.ae2.decoder(dec3)
        out = self.ae1.decoder(dec2)

        return lat, out

def build_new_sae_as_frontend(pretrained=True, freeze=False, lr=1e-3):
    '''
    In pretrained blocks you have to mention indices of autoencoder blocks, 
    which must be pretrained, for the others we will initialize their weights randomly
    '''
    model = SAE()
    if pretrained:
        print('[INFO]: Loading pretrained weights for SAE')
        checkpoint = torch.load('./sae_weights/ae123_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print('[INFO]: NOT loading pretrained weights for SAE')
    
    for params in model.parameters():
        params.requires_grad = not freeze
    
    return model
