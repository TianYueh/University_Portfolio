import torch
from tqdm import tqdm
from utils import dice_score

def validate(model, valid_loader, device, criterion):
    valid_loss = 0.0
    valid_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating", ncols=80):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
            preds = model(images)
            loss = criterion(preds, masks)
            valid_loss += loss.item()
            
            preds = torch.sigmoid(preds)
            dice = dice_score(preds, masks)
            valid_dice += dice
            
    valid_loss /= len(valid_loader)
    valid_dice /= len(valid_loader)
    return valid_loss, valid_dice
