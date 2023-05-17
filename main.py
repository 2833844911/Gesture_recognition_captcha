import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1, :, :]
        targets = (targets == 1).float()
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.95, smooth=1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        ce_loss = self.ce_loss(logits, targets)
        combined_loss = self.alpha * dice_loss + (1 - self.alpha) * ce_loss
        return combined_loss, ce_loss

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


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossEn = nn.CrossEntropyLoss()

    def forward(self,da,ta):
        # loss1 = self.crossEn(da,ta)
        dn, ind = torch.max(da, dim=1)
        b = torch.sum(dn[(ind == 1) & (ta == 1)])
        bm = torch.sum(  dn[(ind == 1) | (ta == 1)])
        diss = 1 - (b)/bm

        b = torch.sum(dn[(ind == 0) & (ta == 0)])
        bm = torch.sum(dn[(ind == 0) | (ta == 0)])
        diss2 = 1 - (b) / bm

        # loss1 +=( diss + diss2)
        loss1 =( diss + diss2)

        return loss1, diss+diss2


# 生成数据
class CustomDataset(Dataset):
    def __init__(self):
        self.dataFiles = os.listdir('./imgs/target')
        self.length = len(self.dataFiles)

    def __getitem__(self, index):
        fname = self.dataFiles[index]
        image = cv2.imread('./imgs/target/'+fname)
        image = cv2.resize(image, (280, 176))
        mask = cv2.imread('./imgs/label/'+fname.replace('.jpg', '_pseudo.png'), 0)
        mask = cv2.resize(mask, (280, 176))
        mask[mask>0] = 1
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.permute(image, (2, 0, 1))
        return image, torch.tensor(mask, dtype=torch.long)
    def __len__(self):
        return self.length

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loader = tqdm(train_loader)
    allloss = 0
    mogg = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        fm = target
        output = torch.softmax(output, dim=1)
        loss, mmloss = criterion(output, fm)
        loss.backward()
        optimizer.step()
        allloss += loss.item()
        mogg += mmloss.item()

        train_loader.set_description(desc='Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} mfloss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader),
            100. * batch_idx / len(train_loader), allloss/(batch_idx+1), mogg/(batch_idx+1) ))

def main():

    model = UNet(3, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # criterion = MyLoss()
    # criterion = DiceLoss()
    criterion = CombinedLoss()

    # 数据集和数据加载器
    train_dataset = CustomDataset()
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    # 训练过程
    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
    torch.save(model,"./fumx.pth")


if __name__ == '__main__':
    main()

