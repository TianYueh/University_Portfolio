import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image
from tqdm import tqdm

# 自訂 Dataset，讀取資料夾中所有圖片檔案
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        # 僅讀取圖片副檔名為 jpg, jpeg, png 的檔案
        self.image_paths = [
            os.path.join(test_dir, fname)
            for fname in os.listdir(test_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        # 將檔名(不含副檔名)存成清單
        self.image_names = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 傳回圖片及對應的檔名
        return image, self.image_names[idx]

def get_test_loader(data_dir, batch_size):
    test_dir = os.path.join(data_dir, 'test')
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, test_dataset

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 載入模型 (注意模型結構須與訓練時相同)
    model = models.resnext50_32x4d(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 100)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 新增：讀取 train 資料夾以取得 class_to_idx，並建立 idx_to_class 的映射
    train_dir = os.path.join(args.data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    test_loader, test_dataset = get_test_loader(args.data_dir, args.batch_size)

    results = []
    progress_bar = tqdm(test_loader, desc='Inference')
    with torch.no_grad():
        for inputs, image_names in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for name, pred in zip(image_names, preds):
                # 將預測的索引轉成對應的類別名稱
                predicted_class = idx_to_class[int(pred.item())]
                results.append((name, predicted_class))

    # 寫入 CSV 檔案
    with open(args.output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_name', 'pred_label'])
        writer.writerows(results)
    print(f'Results saved to {args.output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Image Classification')
    parser.add_argument('--data_dir', type=str, default='./data', help='資料集路徑')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='模型權重檔案')
    parser.add_argument('--output_csv', type=str, default='prediction.csv', help='輸出CSV檔案名稱')
    args = parser.parse_args()

    main(args)
