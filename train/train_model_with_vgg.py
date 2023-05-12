import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm.auto import tqdm
from tqdm import trange
from model_with_vgg import build_scalenet_vgg
from datasets import create_datasets, create_data_loaders
from utils import save_model, SaveBestModel, custom_loss
from time import sleep
from torchvision import transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=50, help='num of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for our optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of dataloader')
parser.add_argument('-d', '--data', type=str, default='/media/AdditionalDrive/datasets_scale/data_shvarc/', help='data path')
parser.add_argument('-ss', '--sampler', type=int, default=0, help='size of sampler in TrainDataLoader for each label')

args = vars(parser.parse_args())

#GLOBAL VARIABLES
BATCH_SIZE = args['batch_size']
DATA_PATH = args['data']
EPOCHS = args['epochs']
LEARNING_RATE = args['learning_rate']
SAMPLE_SIZE = args['sampler']

# transform = T.Compose([
#     T.ToTensor(),
#     # T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944])
# ])

#CREATE DATASETS FOR TRAINING
train_dataset, valid_dataset, test_dataset = create_datasets(
    data_path=DATA_PATH, transform=True, sample_size_for_each_label=SAMPLE_SIZE)
#CREATE DATALOADERS FOR TRAINING
train_loader, valid_loader, test_loader = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE)
#SHOW ME THE COMPUTATION DEVICE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

#BUILD OUR MODEL WITH VGG_16_BN AS FRONTEND 
model = build_scalenet_vgg(
    pretrained=False,
    pretrained_vgg=True,
    pretrained_back_regr=False,
    freeze_vgg=False,
    freeze_back_regr=False,
    leaky=False
).to(device)
print(model)
checkpoint = torch.load('./weights/model_vgg_self_norm.pth')
model.load_state_dict(checkpoint['model_state_dict'])

#TOTAL AND TRAINABLE PARAMETERS
total_params = sum(p.numel() for p in model.parameters())
print('Total parameters:     ', total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable parameters: ', trainable_params) 

#DEFINE MODEL'S HYPERPARAMETERS
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# opt_sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5) # default factor=0.5
criterion = nn.L1Loss()
# criterion = custom_loss
save_best_model = SaveBestModel()

#checkpoint = torch.load('./weights/model_vgg_aranc_norm.pth')
#model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# opt_sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

def train_epoch(model, trainloader, optimizer, criterion, epoch):
    model.train()
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    shochik = -1
    with tqdm(enumerate(trainloader),total=len(trainloader)) as tepoch:
        for i, data in tepoch:
            tepoch.set_description(f"iter {i}/{len(trainloader)}")
            shochik+=1
            counter += 1
            images, labels = data
            #images = images.permute(0, 3, 1, 2)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1).to(torch.float64)
            # print('outputs: ', outputs.shape, '\n lables: ', labels.shape)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 40 == 0:
                tepoch.set_postfix(loss=train_running_loss/counter)
            if shochik % int((len(trainloader)/20)) == 0:
                with open('./logs/log_model_vgg.txt', 'a') as f:
                    f.write(f'Epoch: {epoch+1} Iter: {shochik/len(trainloader)*100:.0f}% Loss: {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
    
    epoch_loss = train_running_loss / counter
    return epoch_loss


def validate(model, testloader, criterion,epoch):
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
            outputs = outputs.view(-1)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            if counter % 50 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,labels),1)}')
                with open('./logs/log_validation_model_vgg.txt', 'a') as f:
                    f.write(f'Epoch: {epoch} Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
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
        criterion=criterion,
        epoch=epoch,
    )
    valid_epoch_loss = validate(model, valid_loader, criterion,epoch=epoch)

    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    # save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_vgg.pth')
    save_model(epoch, model, optimizer, criterion, path_to_save='./weights/model_vgg_self_norm_sgd.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion, path_to_save='./weights/final_model_vgg_self_norm.pth')
print('TRAINING COMPLETE')

        








