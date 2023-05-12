import onnx
from cv2 import imread
from torchvision import transforms as T
from image_cropping import crop_image_inference
import time
import torch
import onnxruntime as ort

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = onnx.load('scalenet.onnx')

image_path = 'test_image.jpg'
image = imread(image_path)

transforms = T.Compose([
    T.ToTensor(),
    T.CenterCrop(size=(1026, 513)),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]),
    T.Lambda(lambda x: x.to(device))
])
tr_image = transforms(image)
# n = 0
# start_time = time.time()
# while n < 1000:
#     cr_im = crop_image_inference(tr_image)
#     res = model(cr_im)
#     res = torch.mean(res)
#     print(f'{n}: {res}')
#     n += 1
# runtime = time.time() - start_time
# print(runtime / 1000)


cr_im = crop_image_inference(tr_image)
print(cr_im.shape)
ort_sess = ort.InferenceSession('scalenet.onnx')
outputs = ort_sess.run(torch.Tensor, [{'input': cr_im}])
print(outputs)

