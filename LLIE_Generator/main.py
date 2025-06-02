from model import UNet,UNetDeconv, train_transformer, test_transformer
from dataset import ImageDataset 
from trainer import Trainer

import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

root = "D:\\Datasets\\LOLdataset"
model = UNet()
# model.load_state_dict(torch.load("saved\\20250531_163914\\ckpt\\99.pth"))
train_set = ImageDataset(os.path.join(root,"our485"), train_transformer, train_transformer)
test_set = ImageDataset(os.path.join(root,"eval15"), test_transformer, test_transformer)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
loss_fn=nn.L1Loss()
opt=optim.Adam(model.parameters(),lr=1e-5)

trainer = Trainer("cuda", model, opt, loss_fn, train_loader, test_loader, "./saved")

trainer.train(100, 5, 1)