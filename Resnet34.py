import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(           #长连接
            nn.Linear(inchannel, outchannel), 
            #nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),     #2d卷积
            #nn.BatchNorm2d(outchannel),          #正则化，减少过拟合的问题
            #nn.ReLU(inplace=True),     #增加非线性拟合能力
            nn.LeakyReLU(inplace=True), 
            
            nn.Linear(outchannel, outchannel),
            #nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),     #2d卷积
            #nn.BatchNorm2d(outchannel)    
        )
        self.shortcut = nn.Sequential()            #捷径连接
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Linear(inchannel, outchannel),   
                #nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):           #前馈网络
        out = self.left(x)
        out += self.shortcut(x)
        #out = F.relu(out)     #两条路径叠加后加一个relu层
        out = F.leaky_relu(out) 
        return out

class ResNet(nn.Module):      #增加一个原数据到卷积网络的接口
    def __init__(self, ResidualBlock, num_classes=15):   #num_classes在分类问题中指类型数量,Net最后一层为15
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.layer0 = nn.Sequential(
            nn.Linear(4,self.inchannel), 
            #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True）     ori_para: 3，64
            #nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),    #kernel_size卷积核的长度
            #nn.BatchNorm2d(64),   #num_features     会输出num_features个值
            #nn.ReLU(),
            nn.LeakyReLU(), 
            
            
        )                #输入和输出均为64个数据
        self.layer1 = self.make_layer(ResidualBlock, 16,  3, stride=1)    #64->64   64->64   
        self.layer2 = self.make_layer(ResidualBlock, 32, 4, stride=2)    #64->128   128->128 
        self.layer3 = self.make_layer(ResidualBlock, 64, 6, stride=2)     #128->256     256->256
        self.layer4 = self.make_layer(ResidualBlock, 128, 3, stride=2)     # 256->512    512->512  
        self.fc = nn.Linear(128, num_classes)       # 全连接层   512->15
        
        self.dropout = nn.Dropout(p=0.5)   #加一个dropout,防止过拟合

    def make_layer(self, block, channels, num_blocks, stride):    #ResidualBlock, 64,  2, stride=1    ，num_block指定resblock的重复数量
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]    #滑动步长
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))    #block为ResidualBlock(inchannel, outchannel, stride=1)
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)         #卷积层
        out = self.layer1(out)      
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet34():

    return ResNet(ResidualBlock)
