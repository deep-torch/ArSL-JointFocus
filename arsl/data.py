import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from PIL import Image


class KArSL(Dataset):
    def __init__(self, labels_file, paths, max_length, transform=None, target_transform=None):
        self.max_length = max_length
        self.labels = pd.read_excel(labels_file)
        self.items = []

        for path in paths:
            for sign in os.listdir(path):
                # running code on colab sometimes create such folders
                if sign == '.ipynb_checkpoints':
                    continue

                sign_dir = os.path.join(path, sign)

                label_idx = int(sign) - 1
                for item in os.listdir(sign_dir):
                    item_dir = os.path.join(sign_dir, item)
                    self.items.append((label_idx, item_dir))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_idx, item_path = self.items[idx]

        if self.target_transform:
            label = self.target_transform(label)

        video = []
        for frame in os.listdir(item_path):
            path = os.path.join(item_path, frame)
            frame = Image.open(path)

            if self.transform:
                frame = self.transform(frame)

            video.append(frame)

        while len(video) < self.max_length:
              video.append(video[-1])

        tensor = torch.stack(video)
        return tensor, label_idx


def test_train_split(root_dir, mode):
    data_dirs = []
    for signer in sorted(os.listdir(root_dir)):
        if signer == '.ipynb_checkpoints' or signer == 'labels.xlsx':
            continue
        signer_path = os.path.join(root_dir, signer)
        for split in ['train', 'test']:
            split_path = os.path.join(signer_path, split)
            data_dirs.append(split_path)
    
    if mode == "dependent":
        return (
            data_dirs[::2],  # train
            data_dirs[1::2]  # test
        )
    elif mode == "independent":
        train_size = len(data_dirs) - 2
        if train_size == 0:
            raise ValueError("At least 2 signers must be in the dataset to be used in independent mode.")
        return (
            data_dirs[:train_size],  # train
            data_dirs[train_size:]  # test
        )
    else:
        raise ValueError(f'Mode "{mode}" is not supported.'
                          ' Supported modse are: ["dependent", "independent"].')


def get_dataloaders(root_dir, training_mode, model_type, labels_path, batch_size):
    train_dirs, test_dirs = test_train_split(root_dir, training_mode)

    train_transforms = [
        transforms.ToTensor()
    ]

    test_transforms = [
        transforms.ToTensor()
    ]

    if model_type == 'pretrained':
        train_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        test_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    train_data = KArSL(labels_path, train_dirs, max_length=32, transform=train_transforms)
    test_data = KArSL(labels_path, test_dirs, max_length=32, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, test_loader
