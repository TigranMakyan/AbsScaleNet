import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sae_frontend_new import AutoEncoder
import argparse
from datasets import create_data_loaders, create_datasets
from utils import params_count_lite, _initialize_weights, SaveBestModel, save_model
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-ss', '--sampler', type=int, default=0, help='size of sampler in TrainDataLoader for each label')
parser.add_argument('-ep', '--epochs', type=int, default=20, help='num of epochs vor single ae block training')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size for dataloader in single block')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for train single block')
parser.add_argument('-d', '--data', type=str, default='/media/AdditionalDrive/datasets_scale/data_shvarc/', help='data path for training of single block')

args = vars(parser.parse_args())

EPOCHS = args['epochs']
BATCH_SIZE  = args['batch_size']
LEARNING_RATE = args['learning_rate']
DATA = args['data']
SAMPLER_SIZE = args['sampler']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    T.Lambda(lambda x: x.to(device))
])

train_dataset, valid_dataset, test_dataset = create_datasets(data_path = args['data'], \
    transform=transform, sample_size_for_each_label=SAMPLER_SIZE)
# get the training and validaion data loaders
train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE
)

encoder1 = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1, 1), 
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(16),
    nn.LeakyReLU(0.1, True),
    nn.Conv2d(16, 32, 3, 1, 1), 
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1, True)
)
decoder1 = nn.Sequential(
    nn.ConvTranspose2d(32, 16, 3, 1, 1), 
    nn.BatchNorm2d(16),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(16, 8, 5, 2, 1), 
    nn.BatchNorm2d(8),
    nn.LeakyReLU(0.1, True),
    nn.ConvTranspose2d(8, 3, 2, 2, 1),
    nn.BatchNorm2d(3),
    nn.LeakyReLU(0.1, True)
)


save_best_model = SaveBestModel()

ae1 = AutoEncoder(encoder=encoder1, decoder=decoder1).to(device) 
ae1.load_state_dict(torch.load('./sae_weights/ae1_best.pth')['model_state_dict'])
params_count_lite(ae1)

optimizer = optim.AdamW(ae1.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer.load_state_dict(torch.load('./sae_weights/ae1_best.pth')['optimizer_state_dict'])
criterion = nn.MSELoss()

def train_epoch(model, trainloader, optimizer, criterion):
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
            image, _ = data
            image = image.float().to(device)
            optimizer.zero_grad()
            _, outputs = model(image)
            loss = criterion(outputs, image)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 40 == 0:
                tepoch.set_postfix(loss=f'{train_running_loss/counter:.8f}')
            if shochik % int((len(trainloader)/20)) == 0:
                with open('./logs/log_sae.txt', 'a') as f:
                    f.write(f'Iter: {shochik/len(trainloader)*100:.0f}% Loss: {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
    epoch_loss = train_running_loss / counter
    return epoch_loss

def validate_epoch(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, _ = data
            image = image.float().to(device)
            # forward pass
            _, outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, image)
            valid_running_loss += loss.item()
            # if counter % 50 == 0:
                # with open('./logs/log_validation_model_vgg.txt', 'a') as f:
                #     f.write(f'Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss

train_loss, valid_loss = [], []

for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        model=ae1,
        trainloader=train_loader,
        optimizer=optimizer,
        criterion=criterion 
    )                                                                                       
    valid_epoch_loss = validate_epoch(ae1, valid_loader, criterion)

    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    save_best_model(valid_epoch_loss, epoch, ae1, optimizer, criterion, data_path='./sae_weights/ae1_best.pth')
    print('=' * 75)

#SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
save_model(EPOCHS, ae1, optimizer, criterion, path_to_save='./sae_weights/ae1_final.pth')
print('TRAINING COMPLETE')
