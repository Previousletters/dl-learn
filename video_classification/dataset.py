from torch.utils.data import Dataset, DataLoader, Subset
import os
from PIL import Image
import random
import torch
from torch.nn import functional as F
import numpy as np

class VideoDataset(Dataset):
    def __init__(self,root,timesteps,transform):
        self.timesteps=timesteps
        self.transform=transform
        self.root=root
        self.labels=os.listdir(self.root)
        self.dataPaths = []
        self.labels_dict = {}
        for cat in self.labels:
            self.labels_dict[cat] = len(self.labels_dict)
            path2aCat=os.path.join(self.root,cat)
            self.dataPaths += [[os.path.join(path2aCat, i), cat] for i in os.listdir(path2aCat) if os.path.exists(os.path.join(path2aCat, i, "frame0.jpg"))]
        print(self.labels)
        print(len(self.labels_dict))

    def __len__(self):
        return len(self.dataPaths)

    def __getitem__(self,idx):
        path2imgs=[os.path.join(self.dataPaths[idx][0], f"frame{i}.jpg") for i in range(self.timesteps)]
        label=self.labels_dict[self.dataPaths[idx][1]]
        frames=[]
        for p2i in path2imgs:
            frame=Image.open(p2i)
            frames.append(frame)
        seed=np.random.randint(1e9)
        frames_tr=[]
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            frame=self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr=torch.stack(frames_tr)
        return frames_tr, int(label)