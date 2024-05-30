import torch
import torch.nn as nn


# U-Net
def double_conv(in_channels, out_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return layer


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.dlayer1 = double_conv(1, 64)
        self.dlayer2 = double_conv(64, 128)
        self.dlayer3 = double_conv(128, 256)
        self.dlayer4 = double_conv(256, 512)

        self.bottleneck = double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.ulayer4 = double_conv(512+512, 512)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.ulayer3 = double_conv(256+256, 256)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.ulayer2 = double_conv(128+128, 128)
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.ulayer1 = double_conv(64+64, 64)

        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv1 = self.dlayer1(x)
        result = self.maxpool(conv1)

        conv2 = self.dlayer2(result)
        result = self.maxpool(conv2)

        conv3 = self.dlayer3(result)
        result = self.maxpool(conv3)

        conv4 = self.dlayer4(result)
        result = self.maxpool(conv4)

        result = self.bottleneck(result)

        result = self.upconv4(result)
        result = torch.cat([result, conv4], dim=1)
        result = self.ulayer4(result)

        result = self.upconv3(result)
        result = torch.cat([result, conv3], dim=1)
        result = self.ulayer3(result)

        result = self.upconv2(result)
        result = torch.cat([result, conv2], dim=1)
        result = self.ulayer2(result)

        result = self.upconv1(result)
        result = torch.cat([result, conv1], dim=1)
        result = self.ulayer1(result)

        result = self.final(result)

        return result


# UU-Net (U-Net + V-Net)
class UUNet(nn.Module):
    def __init__(self):
        super(UUNet, self).__init__()

        self.dlayer1 = nn.Sequential(
            single_conv(1, 64),
            single_conv(64, 64)
        )
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.dlayer2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )
        self.result2 = nn.Conv2d(in_channels=128+64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.dlayer3 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )
        self.result3 = nn.Conv2d(in_channels=256+128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            single_conv(256, 512),
            single_conv(512, 512),
            single_conv(512, 512),
            single_conv(512, 512),
            single_conv(512, 512),
            single_conv(512, 512)
        )

        self.up3 = up_conv(512, 256)
        self.uplayer3 = nn.Sequential(
            single_conv(256+256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up2 = up_conv(256+256, 128)
        self.uplayer2 = nn.Sequential(
            single_conv(128+128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up1 = up_conv(128+128, 64)
        self.uplayer1 = nn.Sequential(
            single_conv(64+64, 64),
            single_conv(64, 64),
        )

        self.final = nn.Sequential(
            single_conv(64+64, 64),
            single_conv(64, 1)
        )

    def forward(self, x):
        dlayer1 = self.dlayer1(x)
        down1 = self.down1(dlayer1)

        dlayer2 = self.dlayer2(down1)
        concat2 = torch.cat([dlayer2, down1], dim=1)
        result2 = self.result2(concat2)
        down2 = self.down2(result2)

        dlayer3 = self.dlayer3(down2)
        concat3 = torch.cat([dlayer3, down2], dim=1)
        result3 = self.result3(concat3)
        down3 = self.down3(result3)

        bottleneck = self.bottleneck(down3)

        up3 = self.up3(bottleneck)
        uresult3 = torch.cat([up3, result3], dim=1)
        uplayer3 = self.uplayer3(uresult3)
        new3 = torch.cat([uplayer3, up3], dim=1)

        up2 = self.up2(new3)
        uresult2 = torch.cat([up2, result2], dim=1)
        uplayer2 = self.uplayer2(uresult2)
        new2 = torch.cat([uplayer2, up2], dim=1)

        up1 = self.up1(new2)
        uresult1 = torch.cat([up1, dlayer1], dim=1)
        uplayer1 = self.uplayer1(uresult1)
        new1 = torch.cat([uplayer1, up1], dim=1)

        final = self.final(new1)

        return final


class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(single_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.up_conv(x)

        return x
