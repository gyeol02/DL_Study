import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_ch, bottleneck_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = bottleneck_ch * self.expansion

        self.conv1 = nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_ch)

        self.conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_ch)

        self.conv3 = nn.Conv2d(bottleneck_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        # short cut
        x = self.relu(x + shortcut)
        return x
        
class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes=10):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, bottleneck_ch, blocks, stride=1):
        downsample = None
        out_ch = bottleneck_ch * block.expansion

        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        layers = [block(self.in_ch, bottleneck_ch, stride, downsample)]
        self.in_ch = out_ch

        for _ in range(1, blocks):
            layers.append(block(self.in_ch, bottleneck_ch))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class ResNet50(ResNet):
    def __init__(self, n_classes=10):
        super().__init__(block=BottleneckBlock, layers=[3, 4, 6, 3], n_classes=n_classes)

class ResNet101(ResNet):
    def __init__(self, n_classes=10):
        super().__init__(block=BottleneckBlock, layers=[3, 4, 23, 3], n_classes=n_classes)

class ResNet152(ResNet):
    def __init__(self, n_classes=10):
        super().__init__(block=BottleneckBlock, layers=[3, 8, 36, 3], n_classes=n_classes)






