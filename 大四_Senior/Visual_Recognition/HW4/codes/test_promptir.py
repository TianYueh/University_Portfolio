import os
import torch
import csv
from PIL import Image
from torchvision import transforms
from train_promptir import PromptIR_UNet

def infer_test_set(model_path,
                   test_dir='./hw4_dataset/test/degraded',
                   csv_path='./weather.csv',
                   output_dir='./images',
                   device=None,
                   prompt_dim=16):
    """
    Run inference on the degraded test images using labels defined in a CSV mapping.
    CSV format: <index>,<label> per line, where <index> matches the numeric part of filenames.
    Filenames expected: '[prefix-]<index>.png'; index parsed automatically.
    """
    # set device
    device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(output_dir, exist_ok=True)

    # load label mapping from CSV
    label_map = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                idx = int(row[0])
                lbl = int(row[1])
                label_map[idx] = lbl
            except ValueError:
                continue

    # load model
    model = PromptIR_UNet(prompt_dim=prompt_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # transforms
    to_tensor = transforms.ToTensor()
    to_pil   = transforms.ToPILImage()

    # gather test files
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')],
                   key=lambda x: int(''.join(filter(str.isdigit, x))))

    with torch.no_grad():
        for fname in files:
            path = os.path.join(test_dir, fname)
            img = Image.open(path).convert('RGB')
            inp = to_tensor(img).unsqueeze(0).to(device)

            # parse numeric index from filename
            num = int(''.join(filter(str.isdigit, fname)))
            lbl_val = label_map.get(num, 0)
            label = torch.tensor([lbl_val], dtype=torch.long, device=device)

            # forward
            out = model(inp, label)
            out_img = to_pil(out.clamp(0,1).cpu().squeeze(0))

            # save
            save_name = fname
            out_img.save(os.path.join(output_dir, save_name))
            print(f"Saved denoised {fname} with label={lbl_val} to {output_dir}")

if __name__ == '__main__':
    model_checkpoint = './checkpoints_prompt/best_promptir.pth'
    infer_test_set(model_checkpoint)
