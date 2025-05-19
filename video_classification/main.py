from model import CNNLSTM, CNNTransformer, train_transformer, test_transformer
from dataset import VideoDataset
from trainer import Trainer

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

timesteps = 16
root = "D:\\Datasets\\hmdb51_jpg_lite\\"
model = CNNTransformer()
# model.load_state_dict(torch.load("saved\\20250515_171922\\ckpt\\2.pth"))
train_set = VideoDataset(root+"train", timesteps, train_transformer)
test_set = VideoDataset(root+"test", timesteps, test_transformer)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=1e-5)

trainer = Trainer("cuda", model, opt, loss_fn, train_loader, test_loader, "./saved")

trainer.train(15, 1, 1)