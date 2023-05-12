from torchvision import transforms  as T
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, SubsetRandomSampler
import cv2
import numpy as np
import torch


# data constants
VALID_SPLIT = 0.2
NUM_WORKERS = 0


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.label = float(file_path[-4:])
        self.image_paths = os.listdir(file_path)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        i = os.path.join(self.file_path, image_filepath)
        image = cv2.imread(i)
        # image = image / 255
        # image = image.transpose(2, 0, 1)
        label = self.label
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

class SampledDataset(Dataset):
    def __init__(self, file_path, sample_size=0, transform=None) -> None:
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.label = float(file_path[-4:]) * 100
        self.image_paths_ls = os.listdir(file_path)
        if sample_size==0:
            self.image_paths = self.image_paths_ls
        else:
            indices = list(range(len(self.image_paths_ls)))
            np.random.shuffle(indices)
            indices = indices[:sample_size]
            self.image_paths = [self.image_paths_ls[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        i = os.path.join(self.file_path, image_filepath)
        image = cv2.imread(i)
        label = self.label

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def create_datasets_old(data_path, valid_split=0.2, transform=None):
    """
    Function to build the training, validation, and testing dataset.
    """
    # we choose the `train_dataset` and `valid_dataset` from the same...
    # ... distribution and later one divide 
    train_path = os.path.join(data_path, 'train/')
    test_path = os.path.join(data_path, 'test/')

    paths_tr = [os.path.join(train_path, i) for i in os.listdir(train_path)]
    paths_ts = [os.path.join(test_path, i) for i in os.listdir(test_path)]

    list_of_train_paths = [SampledDataset(file_path=i, transform=transform) for i in paths_tr]
    list_of_test_paths = [SampledDataset(file_path=i, transform=transform) for i in paths_ts]

    train_all_set = ConcatDataset(list_of_train_paths)
    test_set = ConcatDataset(list_of_test_paths)

    count = len(train_all_set)
    test_count = int(valid_split * len(train_all_set))
    train_set, valid_set = random_split(train_all_set, lengths=[count - test_count, test_count])

    print(f"Total training images: {len(train_set)}")
    print(f"Total validation images: {len(valid_set)}")
    print(f"Total test images: {len(test_set)}")

    return train_set, valid_set, test_set


def create_datasets(data_path, transform=None, sample_size_for_each_label=0):
    """
    Function to build the training, validation, and testing dataset.
    Attention======sample size is size of dataset for each label
    """
    train_path = os.path.join(data_path, 'train/')
    valid_path = os.path.join(data_path, 'valid/')
    test_path = os.path.join(data_path, 'test/')

    paths_tr = [os.path.join(train_path, i) for i in os.listdir(train_path)]
    paths_ts = [os.path.join(test_path, i) for i in os.listdir(test_path)]
    paths_val = [os.path.join(valid_path, i) for i in os.listdir(	)]
        
    list_of_train_sets = [SampledDataset(file_path=i, transform=transform, sample_size=sample_size_for_each_label) for i in paths_tr]
    list_of_test_sets = [SampledDataset(file_path=i, transform=transform, sample_size=int(sample_size_for_each_label/10)) for i in paths_ts]
    list_of_valid_sets = [SampledDataset(file_path=i, transform=transform, sample_size=int(sample_size_for_each_label/10)) for i in paths_val]

    train_set = ConcatDataset(list_of_train_sets)
    test_set = ConcatDataset(list_of_test_sets)
    valid_set = ConcatDataset(list_of_valid_sets)

    print(f"Total training images: {len(train_set)}")
    print(f"Total validation images: {len(valid_set)}")
    print(f"Total test images: {len(test_set)}")

    return train_set, valid_set, test_set
    

def create_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size, train_sample_size=None):
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader, test_loader
