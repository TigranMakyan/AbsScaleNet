import torch
import torch.nn as nn
import torch.optim as optim
from model_with_sae import build_scalenet_sae
from utils import params_count_lite, save_model, SaveBestModel
import argparse
from datasets import create_data_loaders, create_datasets
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help='num of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for our optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size of dataloader')
parser.add_argument('-d', '--data', type=str, default='./data/', help='data path')

args = vars(parser.parse_args())

#GLOBAL VARIABLES
BATCH_SIZE = args['batch_size']
DATA_PATH = args['data']
EPOCHS = args['epochs']
LEARNING_RATE = args['learning_rate']

#CREATE DATASETS FOR TRAINING
train_dataset, valid_dataset, test_dataset = create_datasets(data_path=DATA_PATH)
#CREATE DATALOADERS FOR TRAINING
train_loader, valid_loader, test_loader = create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

model = build_scalenet_sae(
    pretrained=False,
    pretrained_back_regr=False,
    pretrained_sae=True, 
    freeze_sae=True, 
    freeze_back_regr=False
).to(device)

#TOTAL AND TRAINABLE PARAMETERS
total_params = sum(p.numel() for p in model.parameters())
print('Total parameters:     ', total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable parameters: ', trainable_params) 

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
criterion = nn.L1Loss()
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion):
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        images, labels = data
        #images = images.permute(0, 3, 1, 2)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.squeeze()
        print('outputs: ', outputs.shape, '\n lables: ', labels.shape)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
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

    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='weights/model_vgg.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion, path_to_save='weighst/final_model_vgg.pth')
print('TRAINING COMPLETE')





