import cv2
import os
import torch
def start_points(size, split_size, overlap=0.5):
    points = []
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def crop_images(img_path, load_path, split_width=256, split_height=256, overlap=0.5, counter=1):
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    count = 0
    name = 'splitted'
    frmt = 'jpg'

    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            cv2.imwrite(load_path + '{}_{}_{}.{}'.format(counter, name, count, frmt), split)
            count += 1
    print('Image is cropped')

def image_sampling(initial_dir, final_dir):
    # directory = '/home/user/Desktop/work/inria/'
    # download_path = '/home/user/Desktop/work/Scaling/data/train/0.15'
    
    counter = 0
    for filename in os.listdir(initial_dir):
        counter += 1
        im_path = os.path.join(initial_dir, filename)
        crop_images(im_path, final_dir, split_width=256, split_height=256, overlap=0.5, counter=counter)
    
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