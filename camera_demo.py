import cv2
from video_utils import VideoCaptureThreading, Screenshot, FPS
from sae_multi_back import build_scalenet_multi
from torchvision import transforms as T
import torch
from image_cropping import start_points
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model = build_scalenet_multi(True, False, False).to(device)
model.eval()
transform = T.Compose([
    T.ToTensor(),
    T.CenterCrop(size=(512, 512)),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]),
    T.Lambda(lambda x:x.to(device)) 
])

def run_model(frame):
    frame = transform(frame)
    cr_im = crop_image_inference(frame)
    res = model(cr_im)
    return float(torch.mean(res))

cap = VideoCaptureThreading(src=0, resolution=(1280, 720), fourcc='MJPG')
cap.start()
fps = FPS()

for f in cap.generator():
    scale = str(run_model(f))
    cv2.putText(f, scale, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(f, str(fps()), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow("f",f)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
cap.stop()


