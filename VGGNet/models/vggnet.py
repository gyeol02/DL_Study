import torch.nn as nn

class LayerBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, block ,layers, n_classes=10):
        super().__init__()
        self.in_ch = 3  

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.layer5 = self._make_layer(block, 512, layers[4])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block ,out_ch, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_ch, out_ch))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)          
        x = x.view(x.shape[0], -1)   
        x = self.fc(x)               
        return x

class VGGNet11(VGGNet):
    def __init__(self, n_classes=10):
        super().__init__(block=LayerBlock, layers=[1, 1, 2, 2, 2], n_classes=n_classes)

class VGGNet13(VGGNet):
    def __init__(self, n_classes=10):
        super().__init__(block=LayerBlock, layers=[2, 2, 2, 2, 2], n_classes=n_classes)

class VGGNet16(VGGNet):
    def __init__(self, n_classes=10):
        super().__init__(block=LayerBlock, layers=[2, 2, 3, 3, 3], n_classes=n_classes)

class VGGNet19(VGGNet):
    def __init__(self, n_classes=10):
        super().__init__(block=LayerBlock, layers=[2, 2, 4, 4, 4], n_classes=n_classes)

