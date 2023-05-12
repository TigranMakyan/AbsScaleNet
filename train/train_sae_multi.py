import torch
import torch.nn as nn
import torch.optim as optim
from sae_multi_back import build_scalenet_multi
from utils import params_count_lite, save_model, SaveBestModel
import argparse
from datasets import create_data_loaders, create_datasets
from tqdm.auto import tqdm
from torchvision import transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help='num of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for our optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size of dataloader')
parser.add_argument('-d', '--data', type=str, default='/media/AdditionalDrive/datasets_scale/data_shvarc/', help='data path for training of single block')
parser.add_argument('-ss', '--sampler_size', type=int, default=0)
args = vars(parser.parse_args())

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

#GLOBAL VARIABLES
BATCH_SIZE = args['batch_size']
DATA_PATH = args['data']
EPOCHS = args['epochs']
LEARNING_RATE = args['learning_rate']
SAMPLER_SIZE = args['sampler_size']

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    T.Lambda(lambda x: x.to(device))
])

#CREATE DATASETS FOR TRAINING
train_dataset, valid_dataset, test_dataset = create_datasets(data_path = args['data'], \
    transform=transform, sample_size_for_each_label=SAMPLER_SIZE)
# get the training and validaion data loaders
train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE
)

model = build_scalenet_multi(
    pretrained=False,
    pretrained_sae=False, 
    freeze_sae=False,
).to(device)
checkpoint = torch.load('./weights/model_multi_gray.pth')
model.load_state_dict(checkpoint['model_state_dict'])

modules = [model.frontend, model.back1, model.back2, model.back3]
for module in modules:
    for p in module.parameters():
        p.requires_grad = False


params_count_lite(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
criterion = nn.L1Loss()
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion, epoch):
    print('TOTAL SAE TRAINING')
    model.train()
    train_running_loss = 0.0
    counter = 0
    shochik = -1
    with tqdm(enumerate(trainloader), total=len(trainloader)) as tepoch:
        for i, data in tepoch:
            tepoch.set_description(f"iter {i}/{len(trainloader)}")
            shochik += 1
            counter += 1
            images, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1).to(torch.float64)
            # print(labels.dtype, outputs.dtype)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 200 == 0:
                tepoch.set_postfix(loss=train_running_loss/counter)
            if shochik % int((len(trainloader)/20)) == 0:
                with open('./logs/log_sae.txt', 'a') as f:
                    f.write(f'Epoch: {epoch} Iter: {shochik/len(trainloader)*100:.0f}% Loss: {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
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
            outputs = outputs.view(-1)
            # calculate the loss
            print(image.dtype)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            if counter % 50 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,labels),1)}')
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
        epoch=epoch
    )
    valid_epoch_loss = validate(model, valid_loader, criterion)

    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='weights/model_multi_FT_MSE.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, model, optimizer, criterion, path_to_save='weighst/final_model_multi.pth')
print('TRAINING COMPLETE')





