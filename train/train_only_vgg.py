import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from vgg_frontend import build_vgg_as_frontend
from utils import _initialize_weights
from vgg_frontend import build_vgg_as_frontend
import argparse
from datasets import create_data_loaders, create_datasets
from utils import save_model, SaveBestModel
from tqdm.auto import tqdm
import torch.optim as optim


parser = argparse.ArgumentParser()
#TOTAL SAE TRAIN HYPERPARAMETERS
parser.add_argument('-t_ep', '--total_epochs', type=int, default=30, help='num of epochs for the ENTIRE sae training')
parser.add_argument('-t_bs', '--total_batch_size', type=int, default=32, help='batch size of total sae dataloader')
parser.add_argument('-t_lr', '--total_learning_rate', type=float, default=1e-5, help='learning rate od sae training')
parser.add_argument('-t_d', '--total_data', type=str, default='./data/', help='data path for training sae')

args = vars(parser.parse_args())

EPOCHS = args['epochs']
BATCH_SIZE  = args['batch_size']
LEARNING_RATE = args['learning_rate']
DATA = args['data']

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('COMPUTATION DEVICE:  ', device)


train_dataset, valid_dataset, test_dataset = create_datasets(data_path = args['data'])
# get the training and validaion data loaders
train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE
)

#CREATE MODEL FROM VGG
modul = vgg16_bn(pretrained=True)
modul.avgpool = nn.AdaptiveAvgPool2d((4, 4))
# print(modul.classifier)
modul.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512*16, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 6),
    nn.Softmax(dim=1)
)
model = nn.Sequential(
    modul.features,
    modul.avgpool,
    modul.classifier
).to(device)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print('Total parameters:     ', total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable parameters: ', trainable_params) 

#DEFINE MODEL'S HYPERPARAMETERS
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
criterion = nn.L1Loss()
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion):
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    with tqdm(enumerate(trainloader),total=len(trainloader)) as tepoch:
        for i, data in tepoch:
            tepoch.set_description(f"iter {i}/{len(trainloader)}")
            counter += 1
            images, labels = data
            #images = images.permute(0, 3, 1, 2)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            # print('outputs: ', outputs.shape, '\n lables: ', labels.shape)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 20 == 0:
                tepoch.set_postfix(loss=loss.item())
    
    epoch_loss = train_running_loss / counter
    return epoch_loss


def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            if counter == 10:
                print('===LOOK these are our labels:=== \n', labels)
                print('===LOOK these are our outputs:=== \n', outputs)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss


train_loss = []
valid_loss = []

#START THE TRAINING

for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        model=model,
        trainloader=train_loader,
        optimizer=optimizer,
        criterion=criterion
    )
    valid_epoch_loss = validate(model, valid_loader, criterion)

    scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='/home/powerx/computer_vision/scalenet/weights/model_vgg.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion, path_to_save='/home/powerx/computer_vision/scalenet/weights/final_model_vgg.pth')
print('TRAINING COMPLETE')




