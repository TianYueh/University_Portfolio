# train_promptir.py
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
# Dataset: returns image, clean, noise_type (0=snow,1=rain)
# ----------------------------
class PromptDataset(Dataset):
    def __init__(self, degraded_files, clean_dir, transform=None):
        self.files = degraded_files
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        name = os.path.basename(deg_path)
        prefix, num = os.path.splitext(name)[0].split('-')
        label = 0 if prefix=='snow' else 1
        clean_name = f"{prefix}_clean-{num}.png"
        clean_path = os.path.join(self.clean_dir, clean_name)
        img = Image.open(deg_path).convert('RGB')
        tgt = Image.open(clean_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            tgt = self.transform(tgt)
        return img, tgt, label

# ----------------------------
# FiLM block: generate gamma, beta from prompt embedding
# ----------------------------
class FiLM(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, in_channels*2)
        )
    def forward(self, x, emb):
        # emb: (B, embed_dim)
        params = self.fc(emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta  = beta.view(-1, x.size(1), 1, 1)
        return gamma * x + beta

# ----------------------------
# UNet with FiLM in each encoder block
# ----------------------------
class PromptIR_UNet(nn.Module):
    def __init__(self, prompt_dim=16):
        super().__init__()
        self.embed = nn.Embedding(2, prompt_dim)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(inplace=True)
        )
        self.film1 = FiLM(64, prompt_dim)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(inplace=True)
        )
        self.film2 = FiLM(128, prompt_dim)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(inplace=True)
        )
        self.film3 = FiLM(256, prompt_dim)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        # decoder
        self.up4 = nn.ConvTranspose2d(512,256,2,2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(64,3,1)

    def forward(self, x, label):
        emb = self.embed(label)  # (B, prompt_dim)
        # encode
        e1 = self.enc1(x)
        e1 = self.film1(e1, emb)
        e2 = self.enc2(self.pool(e1))
        e2 = self.film2(e2, emb)
        e3 = self.enc3(self.pool(e2))
        e3 = self.film3(e3, emb)
        e4 = self.enc4(self.pool(e3))
        # decode
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4,e3],1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3,e2],1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2,e1],1))
        return self.final(d2)

# ----------------------------
# PSNR & Loss
# ----------------------------

def compute_psnr(pred, target):
    pred_np = pred.clamp(0,1).cpu().detach().numpy()
    tgt_np = target.cpu().numpy()
    psnrs = [compare_psnr(np.transpose(tgt_np[i],(1,2,0)), np.transpose(pred_np[i],(1,2,0)), data_range=1.0)
             for i in range(pred_np.shape[0])]
    return sum(psnrs)/len(psnrs)

def mixed_loss(pred, target, alpha=0.5):
    return alpha*nn.L1Loss()(pred,target) + (1-alpha)*nn.MSELoss()(pred,target)

# ----------------------------
# Training
# ----------------------------
if __name__=='__main__':
    train_dir = './hw4_dataset/train'
    deg_dir = os.path.join(train_dir,'degraded')
    cln_dir = os.path.join(train_dir,'clean')
    os.makedirs('./checkpoints_prompt', exist_ok=True)

    files = sorted(glob(os.path.join(deg_dir,'*.png')),
                   key=lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[1]))
    split = int(0.8*len(files))
    train_files, val_files = files[:split], files[split:]

    tfm = transforms.ToTensor()
    train_ds = PromptDataset(train_files, cln_dir, tfm)
    val_ds   = PromptDataset(val_files,   cln_dir, tfm)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = PromptIR_UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_psnr = 0
    epochs = 500
    for ep in range(1, epochs+1):
        model.train();
        for x,y,lbl in tqdm(train_loader, desc=f'Epoch{ep} Train'):
            x,y,lbl = x.to(device), y.to(device), lbl.to(device)
            pred = model(x, lbl)
            loss = mixed_loss(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); total_psnr = 0
        with torch.no_grad():
            for x,y,lbl in tqdm(val_loader, desc=f'Epoch{ep} Val'):
                x,y,lbl = x.to(device), y.to(device), lbl.to(device)
                pred = model(x, lbl)
                total_psnr += compute_psnr(pred, y)*x.size(0)
        psnr = total_psnr / len(val_ds)
        print(f'Epoch {ep}: Val_PSNR={psnr:.2f} dB')
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), './checkpoints_prompt/best_promptir.pth')
