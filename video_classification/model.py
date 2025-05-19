import torch
from torch import nn
from einops import rearrange, repeat
import torchvision.transforms as transforms

mean=[0.43216,0.394666,0.37645]
std=[0.22803,0.22145,0.216989]
train_transformer=transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
                    ])

test_transformer=transforms.Compose([
				transforms.Resize((32,32)),
				transforms.ToTensor(),
				transforms.Normalize(mean,std),
				])

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


class CNNLSTM(nn.Module):

    def __init__(self):
        super(CNNLSTM,self).__init__()
        self.backbone = VGG16Backbone()
        self.rnn=nn.LSTM(512, 512, 2, batch_first=True)
        self.fc1=nn.Linear(512, 12)

    def forward(self, x):
        b_z,ts,c,h,w=x.shape
        ii=0
        y=self.backbone((x[:,ii]))
        y = y.view(b_z, -1)
        output,(hn,cn)=self.rnn(y.unsqueeze(1))
        for ii in range(1,ts):
            y=self.backbone((x[:,ii]))
            y = y.view(b_z, -1)
            out,(hn,cn)=self.rnn(y.unsqueeze(1),(hn,cn))
        out=self.fc1(out[:,0])
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class CNNTransformer(nn.Module):

    def __init__(self):
        super(CNNTransformer,self).__init__()
        self.backbone = VGG16Backbone()
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.transformer=Transformer(512, 2, 8, 64, 768)
        self.fc1=nn.Linear(512, 12)

    def forward(self, x):
        b_z,ts,c,h,w=x.shape
        ii=0
        out = []
        for ii in range(0,ts):
            y=self.backbone((x[:,ii]))
            y = y.view(b_z, -1)
            out.append(y)
        x = torch.stack(out, dim=1)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b_z)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(ts + 1)]
        out = self.transformer(x)
        out=self.fc1(out[:,0])
        return out