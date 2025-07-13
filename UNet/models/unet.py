import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Up_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.conv1 = Conv_Block(in_ch, out_ch=64)
        self.conv2 = Conv_Block(in_ch=64, out_ch=128)
        self.conv3 = Conv_Block(in_ch=128, out_ch=256)
        self.conv4 = Conv_Block(in_ch=256, out_ch=512)
        self.conv5 = Conv_Block(in_ch=512, out_ch=1024)

        self.up5 = Up_Conv_Block(in_ch=1024, out_ch=512)
        self.up_conv5 = Conv_Block(in_ch=1024, out_ch=512)

        self.up4 = Up_Conv_Block(in_ch=512, out_ch=256)
        self.up_conv4 = Conv_Block(in_ch=512, out_ch=256)

        self.up3 = Up_Conv_Block(in_ch=256, out_ch=128)
        self.up_conv3 = Conv_Block(in_ch=256, out_ch=128)

        self.up2 = Up_Conv_Block(in_ch=128, out_ch=64)
        self.up_conv2 = Conv_Block(in_ch=128, out_ch=64)

        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        # Decoder
        d5 = self.up5(x5)
        d5 = self.up_conv5(torch.cat([x4, d5], dim=1))

        d4 = self.up4(d5)
        d4 = self.up_conv4(torch.cat([x3, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.up_conv3(torch.cat([x2, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.up_conv2(torch.cat([x1, d2], dim=1))

        x = self.final_conv(d2)
        
        return x