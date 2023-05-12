import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm.auto import tqdm
from torchvision import transforms as T

from utils import SaveBestModel, save_model, params_count_lite, get_lr
from datasets import create_data_loaders, create_datasets
from model_cbam import build_cbam_model
# from torch_snippets import Report

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help='Epochs for training')
parser.add_argument('-bs', '--batch', type=int, default=32, help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Optimizer\'s learning rate')
parser.add_argument('-d', '--data', type=str, default='/data/', help='data path for training')

args = vars(parser.parse_args())
EPOCHS = args['epochs']
BATCH_SIZE = args['batch']
LR = args['learning_rate']
DATA_PATH = args['data']

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.465, 0.472, 0.417], std=[0.505, 0.506, 0.460])
])

train_dataset, valid_dataset, test_dataset = create_datasets(data_path=DATA_PATH, transform=transform)
#CREATE DATALOADERS FOR TRAINING
train_loader, valid_loader, test_loader = create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

model = build_cbam_model(pretrained=False, freeze_resnet=False, freeze_back_regr=False)
params_count_lite(model)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=2)
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion):
    model.train()
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    with tqdm(enumerate(trainloader),total=len(trainloader)) as tepoch:
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            images, labels = data
            #images = images.permute(0, 3, 1, 2)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1).to(torch.float64)
            
            # print(f'outputs {torch.stack((outputs,labels),1)}')
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print('Loss is NaN')
                break
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 100 == 0:
                tepoch.set_postfix(loss=loss.item())
                print(f'outputs {torch.stack((outputs,labels),1)}')
    epoch_loss = train_running_loss / counter
    return epoch_loss

def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        with tqdm(enumerate(testloader),total=len(testloader)) as tepoch:
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)
                # forward pass
                outputs = model(image)
                outputs = outputs.view(-1)
                if counter % 500 == 0:
                    print(f'outputs|labels\n {torch.stack((outputs,labels),1)}')
                # calculate the loss
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss


train_loss = []
valid_loss = []

#START THE TRAINING
# log = Report(EPOCHS)
for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        model=model,
        trainloader=train_loader,
        optimizer=optimizer,
        criterion=criterion
    )
    lr_tr = get_lr(optimizer)
    # log.record(epoch, trn_loss=train_epoch_loss, learning_rate=lr_tr, end='\r')
    valid_epoch_loss = validate(model, valid_loader, criterion)
    # log.record(epoch, val_loss=valid_epoch_loss, end='\r')
    with open('log_cbam.txt', 'a') as f:
        f.write(f'Epoch: {epoch}, tr_loss: {train_epoch_loss}, val_loss: {valid_epoch_loss} \n')
    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_cbam.pth')
    print('=' * 75)
    # log.report_avgs(epoch)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion, path_to_save='./weighst/final_model_cbam.pth')
print('TRAINING COMPLETE')


