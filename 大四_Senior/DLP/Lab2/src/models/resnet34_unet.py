
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 後續層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
    上採樣 + DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        # 使用 ConvTranspose2d 進行反捲積上採樣
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenation
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet34UNet, self).__init__()
        self.resnet = ResNet34()

        #    (conv1 + bn1 + relu + maxpool) 視為 layer0
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1   # 64 channels
        self.layer2 = self.resnet.layer2   # 128 channels
        self.layer3 = self.resnet.layer3   # 256 channels
        self.layer4 = self.resnet.layer4   # 512 channels

        # 3) Decoder: 使用 Up 模組 (ConvTranspose2d + DoubleConv)
        self.up1 = Up(in_channels=512, skip_channels=256, out_channels=256)  # 對應 layer3 skip
        self.up2 = Up(in_channels=256, skip_channels=128, out_channels=128)  # 對應 layer2 skip
        self.up3 = Up(in_channels=128, skip_channels=64,  out_channels=64)   # 對應 layer1 skip
        self.up4 = Up(in_channels=64,  skip_channels=64,  out_channels=64)   # 對應 layer0 skip

        # 4) 最終輸出層：1x1 conv -> num_classes
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        original_sz = x.shape[2:]
        # Encoder
        x0 = self.layer0(x)   # shape: (N, 64, H/4, W/4)
        x1 = self.layer1(x0)  # shape: (N, 64, H/4, W/4)
        x2 = self.layer2(x1)  # shape: (N,128, H/8, W/8)
        x3 = self.layer3(x2)  # shape: (N,256, H/16, W/16)
        x4 = self.layer4(x3)  # shape: (N,512, H/32, W/32)

        # Decoder (U-Net skip connections)
        d1 = self.up1(x4, x3)  # in: (512, 256), out: 256
        d2 = self.up2(d1, x2)  # in: (256, 128), out: 128
        d3 = self.up3(d2, x1)  # in: (128, 64),  out: 64
        d4 = self.up4(d3, x0)  # in: (64, 64),   out: 64

        # Output
        out = self.out_conv(d4)
        out = F.interpolate(out, size=original_sz, mode='bilinear', align_corners=False)
        return out

if __name__ == "__main__":
    model = ResNet34UNet(num_classes=1)
    x = torch.randn(1, 3, 224, 224)  
    y = model(x)
    print("Output shape:", y.shape)

