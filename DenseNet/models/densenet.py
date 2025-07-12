import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense_Layer(nn.Module):
    def __init__(self, in_ch, growth_rate, bottleneck_factor=4):
        super().__init__()
        inter_ch = bottleneck_factor * growth_rate

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class Dense_Block(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate=32):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            layers.append(Dense_Layer(ch, growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_ch = ch  

    def forward(self, x):
        return self.block(x)


class Transition_Layer(nn.Module):
    def __init__(self, in_ch, theta=0.5):
        super().__init__()
        out_ch = int(in_ch * theta)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_ch = out_ch

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, layers, growth_rate=32, init_ch=64, n_classes=10, theta=0.5):
        super().__init__()
        self.growth_rate = growth_rate
        self.init_ch = init_ch
        self.theta = theta

        self.conv1 = nn.Conv2d(3, init_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(init_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_ch = init_ch
        for i, num_layers in enumerate(layers):
            dense_block = Dense_Block(num_layers, num_ch, growth_rate)
            self.blocks.append(dense_block)
            num_ch = dense_block.out_ch
            if i != len(layers) - 1:
                transition = Transition_Layer(num_ch, theta)
                self.transitions.append(transition)
                num_ch = transition.out_ch

        self.bn_final = nn.BatchNorm2d(num_ch)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_ch, n_classes)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseNet121(DenseNet):
    def __init__(self, n_classes=10):
        super().__init__(layers=[6, 12, 24, 16], growth_rate=32, init_ch=64, n_classes=n_classes)

class DenseNet169(DenseNet):
    def __init__(self, n_classes=10):
        super().__init__(layers=[6, 12, 32, 32], growth_rate=32, init_ch=64, n_classes=n_classes)

class DenseNet201(DenseNet):
    def __init__(self, n_classes=10):
        super().__init__(layers=[6, 12, 48, 32], growth_rate=32, init_ch=64, n_classes=n_classes)

class DenseNet264(DenseNet):
    def __init__(self, n_classes=10):
        super().__init__(layers=[6, 12, 64, 48], growth_rate=32, init_ch=64, n_classes=n_classes)