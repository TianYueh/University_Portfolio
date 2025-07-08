# test.py
import os
import torch
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import csv
from train_modify import UNet

def load_weather_map(csv_path='weather.csv'):
    """
    Load CSV with two columns: id,label (1 for rain, 0 for snow)
    Returns dict[int, int]
    """
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                idx = int(row[0])
                label = int(row[1])
                mapping[idx] = label
            except ValueError:
                continue  # skip header or invalid rows
    return mapping


def main():
    # prepare output
    out_dir = './images_weather'
    os.makedirs(out_dir, exist_ok=True)

    # load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_s = UNet().to(device)
    model_r = UNet().to(device)
    model_s.load_state_dict(torch.load('./checkpoints_weather/best_snow.pth', map_location=device))
    model_s.eval()
    model_r.load_state_dict(torch.load('./checkpoints_weather/best_rain.pth', map_location=device))
    model_r.eval()

    # load weather mapping
    weather_map = load_weather_map('weather.csv')

    tfm = transforms.ToTensor()
    src = './hw4_dataset/test/degraded'
    files = sorted(glob(os.path.join(src, '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for fp in tqdm(files, desc='Testing'):
        # parse index from filename
        idx = int(os.path.splitext(os.path.basename(fp))[0])
        label = weather_map.get(idx, 0)  # default to snow if missing
        img = Image.open(fp).convert('RGB')
        inp = tfm(img).unsqueeze(0).to(device)
        # label 1 => rain, 0 => snow
        model = model_r if label == 1 else model_s
        with torch.no_grad():
            pred = model(inp)
        out_arr = (pred.clamp(0,1).cpu().squeeze().permute(1,2,0).numpy() * 255).astype(np.uint8)
        Image.fromarray(out_arr).save(os.path.join(out_dir, os.path.basename(fp)))

    print(f"Done. Processed {len(files)} images.")

if __name__ == '__main__':
    main()
