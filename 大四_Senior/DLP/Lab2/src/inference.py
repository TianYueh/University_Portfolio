import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm  # 進度條

from oxford_pet import load_dataset
from models.unet import UNet
from utils import dice_score
from models.resnet34_unet import ResNet34UNet


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_path', default='./saved_models/model.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--save_dir', type=str, default='./', help='path to the output directory')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], default='unet', help='model to use: unet or resnet34_unet')

    return parser.parse_args()

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'unet':
        model = UNet(in_channels=3, num_classes=1).to(device)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34UNet(num_classes=1).to(device)
    else:
        raise ValueError("Model not implemented, choose either 'unet' or 'resnet34_unet'")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dataset = load_dataset(args.data_path, mode="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)

    total_dice_score = 0.0
    num_samples = 0

    # Inference with tqdm progress bar
    for i, batch in enumerate(tqdm(loader, desc="Running Inference", ncols=80)):
        images = batch["image"].float().to(device)
        masks = batch["mask"].float().to(device) 
        with torch.no_grad():
            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

        # Dice Score 計算
        batch_dice = dice_score(preds, masks)
        total_dice_score += batch_dice * images.size(0)  # 累積所有樣本的 Dice Score
        num_samples += images.size(0)

        for b in range(images.size(0)):
            pred_mask = (preds[b, 0]).cpu().numpy().astype(np.uint8) * 255
            out_pil = Image.fromarray(pred_mask)
            out_pil.save(os.path.join(args.save_dir, f"pred_mask_{i*args.batch_size + b}.png"))

    avg_dice_score = total_dice_score / num_samples if num_samples > 0 else 0
    print(f"Predictions saved to {args.save_dir}")
    print(f"Final Average Dice Score: {avg_dice_score:.4f}")

if __name__ == '__main__':
    args = get_args()
    run_inference(args)
