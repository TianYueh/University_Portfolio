import os
import csv
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import cv2
import tifffile as sio
from torch.cuda.amp import autocast, GradScaler
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# ─── CSV LOG SETUP ───────────────────────────────────────────────────────────────
log_path = "training_log.csv"
if not os.path.exists(log_path):
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

# Dataset definition
class CellDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, resize_size=512):
        self.root_dir = root_dir
        self.ids = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.resize_size = resize_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        folder = os.path.join(self.root_dir, uid)
        image_id = int(uid.replace('-', ''), 16) % (2**31 - 1)

        # Read and convert image
        image = cv2.imread(os.path.join(folder, 'image.tif'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        image = F.resize(image, [self.resize_size, self.resize_size])

        masks, labels = [], []
        for class_id in range(1, 5):
            mask_file = os.path.join(folder, f'class{class_id}.tif')
            if os.path.exists(mask_file):
                class_mask = sio.imread(mask_file)
                class_mask = cv2.resize(
                    class_mask,
                    (self.resize_size, self.resize_size),
                    interpolation=cv2.INTER_NEAREST
                )
                num_labels, label_map = cv2.connectedComponents(class_mask.astype(np.uint8))
                for inst_id in range(1, num_labels):
                    inst_mask = (label_map == inst_id)
                    if np.count_nonzero(inst_mask) == 0:
                        continue
                    masks.append(torch.as_tensor(inst_mask, dtype=torch.uint8))
                    labels.append(class_id)

        boxes, valid_masks, valid_labels = [], [], []
        for m, cls in zip(masks, labels):
            pos = torch.where(m)
            if pos[0].numel() == 0:
                continue
            xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
            ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            valid_masks.append(m)
            valid_labels.append(cls)

        if not boxes:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.resize_size, self.resize_size), dtype=torch.uint8),
                "image_id": torch.tensor([image_id]),
                "area": torch.tensor([]),
                "iscrowd": torch.tensor([]),
            }
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.stack(valid_masks)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([image_id]),
                "area": area,
                "iscrowd": iscrowd,
            }
        return image, target

# collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Model setup
def get_model(num_classes, resize_size=512):
    backbone = resnet_fpn_backbone('resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
    model = MaskRCNN(backbone, num_classes=num_classes)
    model.transform = GeneralizedRCNNTransform(
        min_size=(resize_size,), max_size=resize_size,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
    )
    return model

# Paths and data loaders
DATA_DIR = './hw3-data-release'
train_dir = os.path.join(DATA_DIR, 'train')
resize_size = 1024  # fixed size for all images
full_ds = CellDataset(train_dir, resize_size=resize_size)
val_size = int(0.2 * len(full_ds))
train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=collate_fn, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=collate_fn, pin_memory=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=5, resize_size=resize_size)
model.to(device)
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001, momentum=0.9, weight_decay=5e-4
)
scaler = GradScaler()
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        imgs = [img.to(device) for img in images]
        targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast():
            loss_dict = model(imgs, targs)
            loss = sum(loss for loss in loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    torch.cuda.empty_cache()

    # Validation (using training mode forward under no_grad to get loss)
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            imgs = [img.to(device) for img in images]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            was_training = model.training
            model.train()
            with autocast():
                loss_dict = model(imgs, targs)
            model.train(was_training)
            val_loss += sum(loss for loss in loss_dict.values()).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ─── APPEND TO CSV LOG ───────────────────────────────────────────────────────
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best_model.pth")
