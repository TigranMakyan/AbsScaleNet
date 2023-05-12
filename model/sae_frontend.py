# import torch
# import torch.nn as nn

# from utils import _initialize_weights

# encoder1 = nn.Sequential(
#     nn.Conv2d(3, 16, 3, 1, 1), 
#     nn.MaxPool2d(2, 2),
#     nn.BatchNorm2d(16),
#     nn.LeakyReLU(0.1, True),
#     nn.Conv2d(16, 32, 3, 1, 1), 
#     nn.MaxPool2d(2, 2),
#     nn.BatchNorm2d(32),
#     nn.LeakyReLU(0.1, True)
# )
# decoder1 = nn.Sequential(
#     nn.ConvTranspose2d(32, 16, 3, 1, 1), 
#     nn.BatchNorm2d(16),
#     nn.LeakyReLU(0.1, True),
#     nn.ConvTranspose2d(16, 8, 5, 2, 1), 
#     nn.BatchNorm2d(8),
#     nn.LeakyReLU(0.1, True),
#     nn.ConvTranspose2d(8, 3, 2, 2, 1),
#     nn.BatchNorm2d(3),
#     nn.LeakyReLU(0.1, True)
# )

# encoder2 = nn.Sequential(
#     nn.Conv2d(32, 128, 3, 1, 1),
#     nn.MaxPool2d(2, 2),
#     nn.BatchNorm2d(128),
#     nn.LeakyReLU(0.1, True),
#     nn.Conv2d(128, 256, 3, 1, 1),
#     nn.MaxPool2d(2, 2),
#     nn.BatchNorm2d(256),
#     nn.LeakyReLU(0.1, True)
# )
# decoder2 = nn.Sequential(
#     nn.ConvTranspose2d(256, 128, 3, 1, 1),
#     nn.BatchNorm2d(128),
#     nn.LeakyReLU(0.1, True),
#     nn.ConvTranspose2d(128, 64, 5, 2, 1), 
#     nn.BatchNorm2d(64),
#     nn.LeakyReLU(0.1, True),
#     nn.ConvTranspose2d(64, 32, 2, 2, 1),
#     nn.BatchNorm2d(32),
#     nn.LeakyReLU(0.1, True),
# )
# encoder3 = nn.Sequential(
#     nn.Conv2d(256, 512, 3, 1, 1),
#     nn.MaxPool2d(2, 2),
#     nn.BatchNorm2d(512),
#     nn.LeakyReLU(0.1, True),
#     nn.Conv2d(512, 512, 3, 1, 1), 
#     nn.BatchNorm2d(512),
#     nn.LeakyReLU(0.1, True)
# )
# decoder3 = nn.Sequential(
#     nn.ConvTranspose2d(512, 256, 3, 1, 1),
#     nn.BatchNorm2d(256),
#     nn.LeakyReLU(0.1, True),
#     nn.ConvTranspose2d(256, 256, 4, 2, 1), 
#     nn.BatchNorm2d(256),
#     nn.LeakyReLU(0.1, True),
# )


# class AE12(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.ae1 = AutoEncoder(encoder=encoder1, decoder=decoder1)
#         self.ae2 = AutoEncoder(encoder=encoder2, decoder=decoder2)
    
#     def forward(self, x):
#         enc1 =  self.ae1.encoder(x)
#         lat = self.ae2.encoder(enc1)
#         dec2 = self.ae2.decoder(lat)
#         out = self.ae1.decoder(dec2)

#         return lat, out


# ae2 = AE12().to(device)
# checkpoint2 = torch.load('./sae_weights/ae12_best.pth')
# ae2.load_state_dict(checkpoint2['model_state_dict'])

# class SAE(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.ae12 = ae2
#         for params in self.ae12.parameters():
#             params.requires_grad = False
#         self.ae3 = AutoEncoder(encoder=encoder3, decoder=decoder3)

#     def forward(self, x):
#         enc1 = self.ae12.ae1.encoder(x)
#         enc2 = self.ae12.ae2.encoder(enc1)
#         lat = self.ae3.encoder(enc2)
#         dec3 = self.ae3.decoder(lat)
#         dec2 = self.ae12.ae2.decoder(dec3)
#         out = self.ae12.ae1.decoder(dec2)

#         return lat, out