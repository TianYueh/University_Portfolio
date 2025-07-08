import torch

def dice_score(pred_mask, gt_mask, eps=1e-7):
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    
    if pred_mask.dim() == 3:  # (B, H, W) → (B, 1, H, W)
        pred_mask = pred_mask.unsqueeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))  # Sum over H, W
    total_pixels = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))

    dice_per_sample = (2.0 * intersection + eps) / (total_pixels + eps)  # Prevent division by zero
    return dice_per_sample.mean().item()  # Return the batch average
