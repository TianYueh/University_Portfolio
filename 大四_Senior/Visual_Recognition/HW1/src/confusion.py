import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Define validation transforms
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 請將 'path/to/val' 替換成你的驗證資料路徑
val_dataset = datasets.ImageFolder('./data/val', transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 設定運算設備
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 載入模型架構及權重
model = models.resnext50_32x4d(pretrained=False)  # pretrained=False，因為我們會載入自己的權重
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)  # 根據需要調整類別數
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()  # 設定模型為評估模式

# 進行推論，收集預測與真實標籤
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 計算混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# 視覺化混淆矩陣
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')

# 儲存圖檔
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix image saved as 'confusion_matrix.png'.")
