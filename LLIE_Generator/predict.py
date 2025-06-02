import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import UNet, UNetDeconv, train_transformer, test_transformer
from dataset import ImageDataset 

root = "D:\\Datasets\\LOLdataset"
model = UNet()
model.load_state_dict(torch.load("saved\\20250531_225337\\ckpt\\99.pth"))
# model = UNetDeconv()
# model.load_state_dict(torch.load("saved\\20250531_213924\\ckpt\\99.pth"
train_set = ImageDataset(os.path.join(root,"our485"), train_transformer, train_transformer)
test_set = ImageDataset(os.path.join(root,"eval15"), test_transformer, test_transformer)

train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

data, target = list(test_loader)[3]
model.eval()
result = model(data)

def save(result, name):
    result = result.detach()
    result = (torch.clip(result, -1.0, 1.0)+1.0)*255.0 / 2.0
    result = result.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    print(result.shape)
    image = Image.fromarray(result)
    image.save(name)

save(target, "target.png")
save(result, "result.png")