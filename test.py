# import torch
# from tqdm.auto import tqdm
# from model_with_vgg import build_scalenet_vgg
# from datasets import create_datasets, create_data_loaders

# # computation device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Computation device: {device}\n")

# # create the model
# model = build_scalenet_vgg(pretrained=True)

# # load the best model checkpoint
# best_model_cp = torch.load('weights/model_vgg_3_epochs.pth')
# best_model_epoch = best_model_cp['epoch']
# print(f"Best model was saved at {best_model_epoch} epochs\n")

# # load the last model checkpoint
# # last_model_cp = torch.load('outputs/final_model.pth')
# # last_model_epoch = last_model_cp['epoch']
# # print(f"Last model was saved at {last_model_epoch} epochs\n")

# # get the test dataset and the test data loader
# train_dataset, valid_dataset, test_dataset = create_datasets()
# _, _, test_loader = create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=32)

# # testing function
# def test(model, criterion, testloader):
#     """
#     Function to test the model
#     """
#     # set model to evaluation mode
#     model.eval()
#     print('Testing')
#     valid_running_loss = 0.0
#     counter = 0
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(testloader), total=len(testloader)):
#             counter += 1
            
#             image, labels = data
#             image = image.to(device)
#             labels = labels.to(device)
#             # forward pass
#             outputs = model(image)
#             outputs = torch.squeeze(outputs)
#             print('Labels are: ', labels)
#             print('Outputs are: ', outputs)
#             # calculate the loss
#             loss = criterion(outputs, labels)
#             valid_running_loss += loss.item()
        
#     # loss for the complete epoch
#     epoch_loss = valid_running_loss / counter
#     return epoch_loss


# # test the last epoch saved model
# # def test_last_model(model, checkpoint, test_loader):def test_last_model(model, checkpoint, test_loader):
# #     print('Loading last epoch saved model weights...')
# #     model.load_state_dict(checkpoint['model_state_dict'])
# #     test_loss = test(model, test_loader)
# #     print(f"Last epoch saved model accuracy: {test_loss:.3f}")

# #     print('Loading last epoch saved model weights...')
# #     model.load_state_dict(checkpoint['model_state_dict'])
# #     test_loss = test(model, test_loader)
# #     print(f"Last epoch saved model accuracy: {test_loss:.3f}")

# # test the best epoch saved model
# def test_best_model(model, checkpoint, test_loader):
#     print('Loading best epoch saved model weights...')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     test_loss = test(model, test_loader)
#     print(f"Best epoch saved model accuracy: {test_loss:.3f}")


# if __name__ == '__main__':
#     # test_last_model(model, last_model_cp, test_loader)
#     test_best_model(model, best_model_cp, test_loader)


import torch
from tqdm.auto import tqdm
from model_with_vgg import build_scalenet_vgg
from sae_multi_back import build_scalenet_multi

from datasets import create_datasets, create_data_loaders, create_datasets_old
import numpy as np
from torchvision import transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944])
])
# computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Computation device: {device}\n")

# create the model
model = build_scalenet_multi(pretrained=False).to(device)
# model.eval()
# load the best model checkpoint
best_model_cp = torch.load('./weights/model_multi_gray.pth')
# print('THis is \n',best_model_cp,type(best_model_cp),'\nEnd of state =========================')

best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")

# load the last model checkpoint
# last_model_cp = torch.load('outputs/final_model.pth')
# last_model_epoch = last_model_cp['epoch']
# print(f"Last model was saved at {last_model_epoch} epochs\n")
# get the test dataset and the test data loader
train_dataset, valid_dataset, test_dataset = create_datasets(data_path='/media/AdditionalDrive/datasets_scale/data_zver/', transform=transform)
# train_dataset, valid_dataset, test_dataset = create_datasets(data_path='/home/powerx/computer_vision/scalenet/data/',transform=transform)
_, _, test_loader = create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=32)
# testing function
def test(model, criterion, testloader):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    print('Testing')
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
            # print(f'outputs {torch.concat((outputs,labels),-1)}')
            
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            if counter % 100 == 0:
                print(f'outputs {torch.stack((outputs,labels),1)}')
                print('Loss item', loss.item())
        
    # loss for the complete epoch
    epoch_loss = valid_running_loss / counter
    print('LOSS for epoch',epoch_loss)
    return epoch_loss


# test the last epoch saved model
def test_last_model(model, checkpoint, test_loader):
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = test(model, test_loader)
    print(f"Last epoch saved model accuracy: {test_loss:.3f}")

# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    criterion = torch.nn.L1Loss()
    test_loss = test(model, criterion, test_loader)
    print(f"Best epoch saved model loss: {test_loss:.5f}")

if __name__ == '__main__':
    # test_last_model(model, last_model_cp, test_loader)
    # try:
    #     test_best_model(model, best_model_cp, test_loader)
    # except:    
        # force_cudnn_initialization()
    test_best_model(model, best_model_cp, test_loader)
