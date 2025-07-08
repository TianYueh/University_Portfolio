import argparse
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import random
import numpy as np

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from evaluate import validate  # 從 evaluate.py 匯入驗證函式

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}", flush=True)
    
    print("Loading datasets...", flush=True)
    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")
    print(f"Train Samples: {len(train_dataset)}, Valid Samples: {len(valid_dataset)}", flush=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    if args.model_type == "unet":
        model = UNet(in_channels=3, num_classes=1).to(device)
    elif args.model_type == "resnet34_unet":
        model = ResNet34UNet(num_classes=1).to(device)
    else:
        raise NotImplementedError(f"Model {args.model_type} is not implemented, choose either 'unet' or 'resnet34_unet'.")
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded.", flush=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_dice_score = float('-inf')
    
    print("Start training...", flush=True)
    for e in range(args.epochs):
        epoch_start = time.time()
        model.train()
        run_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {e+1}/{args.epochs}", ncols=80):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
            preds = model(images)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
        
        train_loss = run_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # 呼叫 evaluate.py 裡的 validate 函式進行驗證
        model.eval()
        valid_loss, valid_dice = validate(model, valid_loader, device, criterion)
        print(f"Epoch {e+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}, Time: {epoch_time:.2f}s", flush=True)
        
        if valid_dice > best_dice_score:
            best_dice_score = valid_dice
            torch.save(model.state_dict(), args.model_path)
            print("New Best Model saved.", flush=True)
        
    print("Training finished. Model saved.", flush=True)
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='Path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model_path', type=str, default="./saved_models/model.pth", help='Path to save/load the model')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], default="unet", help='Model to use: unet or resnet34_unet')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
