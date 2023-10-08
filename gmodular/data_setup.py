import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Union
from datasets.YoloDataSet import YoloDataSet

NUM_WORKERS = os.cpu_count()
def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    return create_dataloaders_from_datasets(train_data, test_data, batch_size, num_workers)
   
def create_dataloaders_from_datasets(train_set,
                                     test_set,
                                     batch_size: int,
                                     num_workers: int=NUM_WORKERS):
    class_names = train_set.classes
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    return train_dataloader, test_dataloader, class_names


def create_YoloDatasets(
        train_dir: str,
        test_dir: str,
        transform: transforms):
    return YoloDataSet(train_dir, transform), YoloDataSet(test_dir, transform)


