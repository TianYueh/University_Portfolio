import json
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from objects import OBJECTS_MAP

class IClevrDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path) as f:
            self.conditions = json.load(f)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64,64)),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ColorJitter(0.1,0.1,0.1,0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        self.keys = list(self.conditions.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        fname = self.keys[idx]
        labels = self.conditions[fname]
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        y = torch.zeros(len(OBJECTS_MAP))
        for obj in labels:
            y[OBJECTS_MAP[obj]] = 1
        return x, y

def get_dataloader(json_path, image_dir, batch_size, shuffle=True, num_workers=4):
    ds = IClevrDataset(json_path, image_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
