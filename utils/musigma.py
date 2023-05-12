import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from datasets import create_datasets,create_data_loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_path = '/media/AdditionalDrive/datasets_scale/data_shvarc/train/'
transform_img = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_data = datasets.ImageFolder(
  root=data_path, transform=transform_img
)

batch_size = 100

loader = DataLoader(
  image_data, 
  batch_size = batch_size, 
  num_workers=2)

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    shochik = 0
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        if shochik%200 == 0:
          print(shochik,fst_moment,snd_moment)
        shochik+=1

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  


def get_musigma(loader):
    channels_sum, channels_squared_sum, num_batches, counter = 0, 0, 0, 0

    for data, _ in loader:
        counter += 1
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
        print(counter)          

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


mean, std = get_musigma(loader)
print("mean and std: \n", mean, std)
with open('musigma.txt', 'w') as f:
    f.write('mean: '+str(mean)+'\n'+'std: '+str(std))