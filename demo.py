import cv2
import torch
import torch.nn as nn
import argparse
# from model_with_sae import build_scalenet_sae
from model_with_vgg import build_scalenet_vgg
from sae_multi_back import build_scalenet_multi
from image_cropping import start_points
import numpy as np
from torchvision import transforms as T
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='multi', help='Please choose model\'s backbone. Can be vgg or sae')
parser.add_argument('-p', '--path', type=str, help='You have to input image path')
args = vars(parser.parse_args())

models_type = args['model']
image_path = args['path']

if models_type == 'vgg':
    model = build_scalenet_vgg(pretrained=True)
elif models_type == 'multi':
    print('MULTI_MODEL')
    model = build_scalenet_multi(True, False, True)
else:
    raise AssertionError('Choose correct type of backbone: vgg or sae')

transform = T.Compose([
    T.ToTensor(),
    # T.Grayscale(num_output_channels=3),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    # T.CenterCrop(size=(513, 513))
])

model.eval()
image = cv2.imread(image_path)
tr_image = transform(image)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def crop_image_inference(img, split_width=256, split_height=256, overlap=0):
    
    _, img_h, img_w = img.shape
    
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    images = []
    for i in Y_points:
        for j in X_points:
            split = img[:, i:i+split_height, j:j+split_width]
            images.append(split)
    
    return torch.stack(images, dim=0)


def demo(model, transform, img_path):
    model.eval()

    image = cv2.imread(image_path)
    tr_image = transform(image)
    print(tr_image.device)
    cropped_image = crop_image_inference(tr_image)
    result = model(cropped_image)
    return torch.mean(result)


cr_im = crop_image_inference(tr_image)
# a = cr_im[0].squeeze().permute(1, 2, 0).detach().numpy()
# print(type(a))
# cv2.imshow('image', a)
# cv2.waitKey(0)
out = model(cr_im)
print(out)
print(torch.mean(out))


# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
# res = demo(model, transform, image_path)
# print(res)
# end.record()

# Waits for everything to finish running
# torch.cuda.synchronize()

# print(start.elapsed_time(end))

# i = 0
# times = []
# while i < 1000:
#     start_time = time.time()
#     res = demo(model, transform, image_path)
#     t = time.time() - start_time
#     times.append(t)
#     i += 1
#     print(i)

# times = np.array(times)
# print(np.mean(times))
