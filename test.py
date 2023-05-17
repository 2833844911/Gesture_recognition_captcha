import torch
import torch.nn as nn
from skimage import morphology
import cv2
import os
import numpy as np
import torchvision.transforms as transforms

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder
        self.dec1 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec3 = self.conv_block(128 + 64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.maxpool(x1))
        x3 = self.enc3(self.maxpool(x2))
        x4 = self.enc4(self.maxpool(x3))

        # Decoder with skip connections
        x = self.upsample(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.final_conv(x)
        return x

model = torch.load("./fumx.pth")


def getInfo(path):
    df2 = cv2.imread(path)
    image = cv2.resize(df2, (280, 176))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])(image).to(device)
    image = torch.permute(transform, (2, 0, 1)).unsqueeze(dim=0)
    out = model(image)
    out = torch.argmax(out, dim=1)
    j = torch.zeros((1, 176, 280))
    j[out == 1] = 255

    kp = j.repeat(3, 1, 1)
    kp = torch.permute(kp, (1, 2, 0))

    gj = morphology.skeletonize(np.array( kp.tolist(), dtype=np.uint8))

    # 得到轨迹点
    hn = np.transpose(np.nonzero(gj))
    print(hn)

    # 画图
    es = np.array( kp.tolist(), dtype=np.uint8)
    for i in hn:
        es[i[0],i[1],:] = 100
    cv2.imshow("target", es)
    cv2.imshow("src", df2)
    cv2.waitKey(0)  # 等待用户按下任意键
    cv2.destroyAllWindows()



if __name__ == '__main__':
    getInfo('./imgs/target/37.jpg')


