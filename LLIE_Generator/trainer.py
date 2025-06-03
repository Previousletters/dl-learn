import logging
import time
import os
import sys
import os.path as osp
from model import CNN

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


class Trainer:

    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        optimizer,
        loss_fn,
        train_loader: DataLoader,
        test_loader: DataLoader,
        out,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.out = os.path.join(out, time.strftime("%Y%m%d_%H%M%S"))
        if not osp.exists(self.out):
            os.makedirs(self.out, exist_ok=True)
            os.makedirs(os.path.join(self.out, "ckpt"), exist_ok=True)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.out, './log.log'))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)


    def train(self, epochs, save_interval, eval_interval):
        vgg = CNN()
        vgg_path = os.path.join("..", "image_classification", "saved", "20250518_155537_CNN", "ckpt", "8.pth")
        vgg.load_state_dict(torch.load(vgg_path))
        vgg.eval()
        vgg.to(self.device)
        # transforms.Compose([
        #     transforms.Resize((32,32)),
        #     transforms.ToTensor(),
        # ])
        mse = nn.MSELoss()
        for epoch in range(epochs):
            self.model.train()
            self.logger.info("Epoch = %d" % epoch)
            for step, (x, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                target = target.to(self.device)
                output = self.model(x)
                loss = self.loss_fn(output, target)
                f1 = vgg.backbone(output)
                f2 = vgg.backbone(target)
                f_loss = mse(f1, f2)
                loss += f_loss
                self.logger.debug("[Train %d:%d] loss=%.6f" % (epoch, step, float(loss.detach().cpu())))
                loss.backward()
                self.optimizer.step()

            if epoch % save_interval == save_interval - 1:
                if isinstance(self.model, nn.DataParallel):
                    torch.save(self.model.module.state_dict(), os.path.join(self.out, 'ckpt', "%d.pth" % epoch))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.out, 'ckpt', "%d.pth" % epoch))

            if epoch % eval_interval == eval_interval - 1:
                self.model.eval()
                psnr = 0
                total = 0
                for step, (x, target) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    target = target.to(self.device)
                    output = self.model(x)
                    loss = self.loss_fn(output, target)
                    total += x.shape[0]
                    psnr += 20 * torch.log10(1 / (mse(output, target))) * x.shape[0]
                    self.logger.info("[Test %d:%d] loss=%.6f" % (epoch, step, float(loss.detach().cpu())))
                    f1 = vgg.backbone(output)
                    f2 = vgg.backbone(target)
                    f_loss = mse(f1, f2)
                    self.logger.info("[Test %d:%d] vgg loss=%.6f" % (epoch, step, float(f_loss.detach().cpu())))
                psnr = psnr / total
                self.logger.info("[Test %d:%d] psnr=%.6f" % (epoch, step, float(psnr.detach().cpu())))
                self.optimizer.zero_grad()