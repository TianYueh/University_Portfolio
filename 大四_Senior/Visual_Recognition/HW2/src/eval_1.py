import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import contextlib

# 啟用 cuDNN autotuner
torch.backends.cudnn.benchmark = True

@contextlib.contextmanager
def my_autocast(enabled):
    try:
        with torch.amp.autocast('cuda', enabled=enabled) as cm:
            yield cm
    except TypeError:
        with torch.cuda.amp.autocast(enabled=enabled) as cm:
            yield cm

def collate_fn(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)

class TestDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        # 假設檔名為數字，如 1.png, 2.png, ...，依照檔名排序
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img_id = int(os.path.splitext(file_name)[0])
        img = Image.open(os.path.join(self.img_dir, file_name)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, img_id

    def __len__(self):
        return len(self.img_files)

def get_model(num_classes, checkpoint_path, device):
    model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 讀取 train_json 取得 category mapping資訊用於建立模型時的 num_classes
    with open(args.train_json) as f:
        data = json.load(f)
    # 注意: category id 從 1 開始，所以 num_classes = (數字種類數量) + 1 (背景)
    num_classes = len(data['categories']) + 1

    model = get_model(num_classes, args.checkpoint, device)

    test_ds = TestDataset(os.path.join(args.data_dir, 'test'), transforms=ToTensor())
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    detections = []  # Task 1 的檢測結果
    for images, img_ids in tqdm(test_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            with my_autocast(args.mixed_precision):
                outputs = model(images)
        for output, img_id in zip(outputs, img_ids):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score < args.score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                detections.append({
                    'image_id': img_id,
                    'bbox': [x1, y1, w, h],
                    'score': float(score),
                    'category_id': int(label)
                })

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, 'pred.json')
    with open(pred_path, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"Saved detection results to {pred_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Task1: Evaluate Faster R-CNN for digit detection")
    parser.add_argument('--data_dir', default='nycu-hw2-data', help='dataset root directory')
    parser.add_argument('--train_json', default='nycu-hw2-data/train.json', help='path to train json (for category mapping)')
    parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
    parser.add_argument('--output_dir', default='results', help='directory to save pred.json')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='number of DataLoader workers')
    parser.add_argument('--score_thresh', type=float, default=0.05, help='score threshold for detections')
    parser.add_argument('--mixed_precision', action='store_true', help='use AMP mixed precision for inference')
    args = parser.parse_args()
    main(args)
