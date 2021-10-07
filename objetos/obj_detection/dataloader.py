from torchvision import datasets, transforms
import os
import torch
from dataset import ListDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloader(dataset_path, img_size, batch_size, num_classes=11):
    bs = {'train': batch_size, 'test': 1}
    image_datasets = {x: ListDataset(dataset_path, x, img_size, num_classes=num_classes) for x in ['train', 'test']}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=bs[x], shuffle=True, num_workers=0, drop_last=False,
                        collate_fn=collate_fn) 
                        for x in ['train', 'test']}

    return dataloaders_dict