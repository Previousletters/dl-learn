from torch.utils.data import Dataset, DataLoader, Subset
import os
from PIL import Image
import random
import torch
from torch.nn import functional as F
import numpy as np


import logging
 
logging.getLogger('PIL').setLevel(logging.WARNING)

class ImageDataset(Dataset):
    def __init__(self,root,transform,label_transform):
        self.transform=transform
        self.label_transform=label_transform
        self.root=root
        files = os.listdir(os.path.join(self.root, "high"))
        self.imgs = [i for i in files if ".png" == i[-4:] ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        low=Image.open(os.path.join(self.root, "low", self.imgs[idx]))
        high=Image.open(os.path.join(self.root, "high", self.imgs[idx]))
        low = self.transform(low)
        high = self.label_transform(high)
        return low, high