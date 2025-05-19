from model import CNN, ViT, train_transformer, test_transformer
from trainer import Trainer

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

model = ViT()
model.load_state_dict(torch.load("saved\\20250519_093250\\ckpt\\9.pth"))
train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transformer)
test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transformer)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=3e-5)

trainer = Trainer("cuda", model, opt, loss_fn, train_loader, test_loader, "./saved")

trainer.train(10, 1, 1)