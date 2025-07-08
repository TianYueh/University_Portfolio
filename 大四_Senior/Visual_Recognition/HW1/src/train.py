import os
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def get_data_loaders(data_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)  
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, save_path):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每個 epoch 包含訓練與驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(data_loader, desc=phase)
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc  = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 若驗證階段精度提升，儲存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f'** Model saved at epoch {epoch+1} with acc: {epoch_acc:.4f}')

    print(f'Best val Acc: {best_acc:.4f}')
    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)
    return model

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 建立資料讀取器
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)

    # 使用ResNeXt50作為基礎模型
    model = models.resnext50_32x4d(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)  # 假設有 100 個類別
    model = model.to(device)

    # 若先前有儲存最佳模型，則載入權重繼續訓練
    if os.path.exists(args.save_path):
        print(f'Loading checkpoint from {args.save_path}')
        model.load_state_dict(torch.load(args.save_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # 訓練模型
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader,
                                device, args.epochs, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification with ResNeXt50')
    parser.add_argument('--data_dir', type=str, default='./data', help='資料集路徑')
    parser.add_argument('--epochs', type=int, default=25, help='訓練 epoch 數')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='模型儲存路徑')
    args = parser.parse_args()

    main(args)
