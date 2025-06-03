import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
import torchvision.transforms as transforms

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
size = (128, 128)

train_transformer=transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
				    # transforms.Normalize(mean,std),
                    ])

test_transformer=transforms.Compose([
				transforms.Resize(size),
				transforms.ToTensor(),
				# transforms.Normalize(mean,std),
				])



class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
        self.layer1=nn.Sequential(
            # 32x32
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        self.layer2=nn.Sequential(
            # 16x16
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.layer3=nn.Sequential(
            # 8x8
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.layer4=nn.Sequential(
            # 4x4
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )
        self.layer5=nn.Sequential(
            # 2x2
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )
        self.layer6=nn.Sequential(
            # 4x4
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
        )
        self.layer7=nn.Sequential(
            # 8x8
            ConvBlock(512, 256),
            ConvBlock(256, 128),
        )
        self.layer8=nn.Sequential(
            # 16x16
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        )
        self.layer9=nn.Sequential(
            # 32x32
            ConvBlock(128, 64),
            ConvBlock(64, 64),
        )
        self.r = nn.Conv2d(64, 1, 1)
        self.g = nn.Conv2d(64, 1, 1)
        self.b = nn.Conv2d(64, 1, 1)


    def forward(self, x):

        def upsample_cat(y0, y1):
            y0 = F.interpolate(y0, scale_factor=2, mode="bilinear")
            return torch.cat([y0, y1], dim=1)

        x0 = x
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(upsample_cat(x5, x4))
        x7 = self.layer7(upsample_cat(x6, x3))
        x8 = self.layer8(upsample_cat(x7, x2))
        x9 = self.layer9(upsample_cat(x8, x1))

        r = self.r(x9)
        g = self.g(x9)
        b = self.b(x9)
        
        return torch.cat([r, g, b], dim=1)



class UNetDeconv(nn.Module):

    def __init__(self):
        super(UNetDeconv,self).__init__()
        self.layer1=nn.Sequential(
            # 32x32
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        self.layer2=nn.Sequential(
            # 16x16
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.layer3=nn.Sequential(
            # 8x8
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.layer4=nn.Sequential(
            # 4x4
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )
        self.layer5=nn.Sequential(
            # 2x2
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )
        self.deconv1 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.layer6=nn.Sequential(
            # 4x4
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
        )
        self.deconv2 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.layer7=nn.Sequential(
            # 8x8
            ConvBlock(512, 256),
            ConvBlock(256, 128),
        )
        self.deconv3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.layer8=nn.Sequential(
            # 16x16
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        )
        self.deconv4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.layer9=nn.Sequential(
            # 32x32
            ConvBlock(128, 64),
            ConvBlock(64, 64),
        )
        self.r = nn.Conv2d(64, 1, 1)
        self.g = nn.Conv2d(64, 1, 1)
        self.b = nn.Conv2d(64, 1, 1)


    def forward(self, x):

        def upsample_cat(y0, y1, deconv):
            y0 = deconv(y0)
            return torch.cat([y0, y1], dim=1)

        x0 = x
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(upsample_cat(x5, x4, self.deconv1))
        x7 = self.layer7(upsample_cat(x6, x3, self.deconv2))
        x8 = self.layer8(upsample_cat(x7, x2, self.deconv3))
        x9 = self.layer9(upsample_cat(x8, x1, self.deconv4))

        r = self.r(x9)
        g = self.g(x9)
        b = self.b(x9)
        
        return torch.cat([r, g, b], dim=1)

class VGG16Backbone(nn.Module):

    def __init__(self):
        super(VGG16Backbone,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)   #16x16x64
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #8x8x128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)     #4x4x256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #2x2x512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #1x1x512
        )

        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.backbone = VGG16Backbone()
        self.classifier=nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        y=self.backbone(x)
        y = y.view(y.shape[0], -1)
        out=self.classifier(y)
        return out

