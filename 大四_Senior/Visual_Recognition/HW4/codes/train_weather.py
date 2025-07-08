# train_weather.py
import os
import random
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------
# Dataset Definition
# ----------------------------
class RestorationDataset(Dataset):
    def __init__(self, file_list, clean_dir, transform=None):
        self.files = file_list
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        base = os.path.splitext(os.path.basename(deg_path))[0]
        prefix, num = base.split('-')
        clean_name = f"{prefix}_clean-{num}.png"
        clean_path = os.path.join(self.clean_dir, clean_name)

        deg = Image.open(deg_path).convert('RGB')
        cln = Image.open(clean_path).convert('RGB')
        if self.transform:
            deg = self.transform(deg)
            cln = self.transform(cln)
        return deg, cln

# ----------------------------
# Deeper UNet Model (4 levels)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        # Final conv
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # decoder
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return out

# ----------------------------
# PSNR & Loss
# ----------------------------
def compute_psnr(pred, target):
    pred_np = pred.clamp(0,1).cpu().detach().numpy()
    tgt_np = target.cpu().numpy()
    psnrs = [compare_psnr(np.transpose(tgt_np[i],(1,2,0)),
                          np.transpose(pred_np[i],(1,2,0)), data_range=1.0)
             for i in range(pred_np.shape[0])]
    return sum(psnrs)/len(psnrs)

def mixed_loss(pred, target, alpha=0.5):
    l1 = nn.L1Loss()(pred, target)
    mse = nn.MSELoss()(pred, target)
    return alpha * l1 + (1 - alpha) * mse

# ----------------------------
# Training Loop (split by prefix) with progress bars
# ----------------------------
if __name__=='__main__':
    train_dir = './hw4_dataset/train'
    deg_dir = os.path.join(train_dir,'degraded')
    cln_dir = os.path.join(train_dir,'clean')
    os.makedirs('./checkpoints_weather',exist_ok=True)

    # list all degraded files and split train/val
    all_files = sorted(glob(os.path.join(deg_dir,'*.png')),
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[1]))
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    # split by weather prefix in filenames
    train_snow_files = [f for f in train_files if os.path.basename(f).startswith('snow-')]
    train_rain_files = [f for f in train_files if os.path.basename(f).startswith('rain-')]
    val_snow_files   = [f for f in val_files   if os.path.basename(f).startswith('snow-')]
    val_rain_files   = [f for f in val_files   if os.path.basename(f).startswith('rain-')]

    tfm = transforms.ToTensor()
    loader_snow = DataLoader(RestorationDataset(train_snow_files,cln_dir,tfm), batch_size=4, shuffle=True, num_workers=4)
    loader_rain = DataLoader(RestorationDataset(train_rain_files,cln_dir,tfm), batch_size=4, shuffle=True, num_workers=4)
    val_loader_s = DataLoader(RestorationDataset(val_snow_files,cln_dir,tfm), batch_size=4, shuffle=False, num_workers=4)
    val_loader_r = DataLoader(RestorationDataset(val_rain_files,cln_dir,tfm), batch_size=4, shuffle=False, num_workers=4)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model_s = UNet().to(device)
    model_r = UNet().to(device)
    opt_s = torch.optim.Adam(model_s.parameters(), lr=1e-4)
    opt_r = torch.optim.Adam(model_r.parameters(), lr=1e-4)

    best_s, best_r = 0, 0
    epochs = 500
    for ep in range(1, epochs+1):
        # snow training
        model_s.train(); total_loss_s = 0
        for x,y in tqdm(loader_snow, desc=f'Epoch{ep} Snow Train'):
            x,y = x.to(device), y.to(device)
            pred = model_s(x)
            l = mixed_loss(pred, y)
            opt_s.zero_grad(); l.backward(); opt_s.step()
            total_loss_s += l.item() * x.size(0)
        # rain training
        model_r.train(); total_loss_r = 0
        for x,y in tqdm(loader_rain, desc=f'Epoch{ep} Rain Train'):
            x,y = x.to(device), y.to(device)
            pred = model_r(x)
            l = mixed_loss(pred, y)
            opt_r.zero_grad(); l.backward(); opt_r.step()
            total_loss_r += l.item() * x.size(0)
        # snow validation
        model_s.eval(); ps_s = 0
        for x,y in tqdm(val_loader_s, desc=f'Epoch{ep} Snow Val'):
            ps_s += compute_psnr(model_s(x.to(device)), y.to(device)) * x.size(0)
        # rain validation
        model_r.eval(); ps_r = 0
        for x,y in tqdm(val_loader_r, desc=f'Epoch{ep} Rain Val'):
            ps_r += compute_psnr(model_r(x.to(device)), y.to(device)) * x.size(0)

        ns = len(val_snow_files)
        nr = len(val_rain_files)
        ps_s = ps_s / ns if ns > 0 else 0
        ps_r = ps_r / nr if nr > 0 else 0
        print(f'Epoch {ep}: Snow_PSNR={ps_s:.2f}, Rain_PSNR={ps_r:.2f}')

        if ps_s > best_s:
            best_s = ps_s
            torch.save(model_s.state_dict(), './checkpoints_weather/best_snow.pth')
        if ps_r > best_r:
            best_r = ps_r
            torch.save(model_r.state_dict(), './checkpoints_weather/best_rain.pth')
