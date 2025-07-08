import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomAffine, Normalize, RandomErasing
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import contextlib

# 啟用 cuDNN autotuner 加速運算
torch.backends.cudnn.benchmark = True

# 定義一個 autocast 的上下文管理器，嘗試新 API，若失敗則使用舊 API
@contextlib.contextmanager
def my_autocast(enabled):
    try:
        # 新版 API (PyTorch 1.10+ 若有支援)
        with torch.amp.autocast('cuda', enabled=enabled) as cm:
            yield cm
    except TypeError:
        # 若不支援則使用舊版 API
        with torch.cuda.amp.autocast(enabled=enabled) as cm:
            yield cm

def collate_fn(batch):
    return tuple(zip(*batch))

class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)  # 利用 COCO API 載入 JSON 格式的標籤
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.imgs[img_id]['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        boxes, labels, area, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']  # 原始格式為 [x_min, y_min, w, h]
            boxes.append([x, y, x + w, y + h])  # 轉換成模型需要的 [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
            area.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.ids)

def get_model(num_classes, freeze_backbone=False):
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = fasterrcnn_resnet50_fpn_v2(
        pretrained=True,
        anchor_generator=anchor_generator
    )
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = SummaryWriter(log_dir=args.log_dir)

    # 資料增強設定
    train_transforms = Compose([
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        RandomAffine(degrees=2, translate=(0.02, 0.02)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(p=0.5)
    ])
    val_transforms = Compose([ToTensor()])

    train_ds = COCODetectionDataset(
        img_dir=os.path.join(args.data_dir, 'train'),
        ann_file=args.train_json,
        transforms=train_transforms
    )
    val_ds = COCODetectionDataset(
        img_dir=os.path.join(args.data_dir, 'valid'),
        ann_file=args.val_json,
        transforms=val_transforms
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn
    )

    num_classes = len(train_ds.coco.cats) + 1  # 注意: category id 從 1 開始，外加背景
    model = get_model(num_classes, freeze_backbone=args.freeze_backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.mixed_precision)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with my_autocast(args.mixed_precision):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (pbar.n + 1))
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # 為了計算 loss，驗證階段也使用 model.train() 狀態 (但在 torch.no_grad() 下)
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with my_autocast(args.mixed_precision):
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.flush()

        scheduler.step()
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Saved checkpoint: {ckpt_path}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Faster R-CNN ResNet50 v2 on digit dataset")
    parser.add_argument('--data_dir', default='nycu-hw2-data', help='dataset root')
    parser.add_argument('--train_json', default='nycu-hw2-data/train.json', help='path to train json')
    parser.add_argument('--val_json', default='nycu-hw2-data/valid.json', help='path to val json')
    parser.add_argument('--output_dir', default='checkpoints', help='where to save checkpoints')
    parser.add_argument('--log_dir', default='runs', help='where to save tensorboard logs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', default='', help='path to checkpoint to resume')
    parser.add_argument('--mixed_precision', action='store_true', help='use AMP mixed precision')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone to speed up training')
    args = parser.parse_args()
    main(args)
