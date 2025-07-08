import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image

# Dataset for validation (COCO format)
class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_path = self.coco.imgs[img_id]['file_name']
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] - 1)  # zero-indexed
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return img, target

    def __len__(self):
        return len(self.ids)

# Compute IoU between one box and array of boxes
def box_iou_np(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / union

# Load model from checkpoint
def load_model(checkpoint_path, num_classes, device):
    model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model

# Main evaluation and confusion matrix plotting
def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Prepare validation dataset and loader
    val_dataset = COCODetectionDataset(
        img_dir=os.path.join(args.data_dir, 'valid'),
        ann_file=args.val_json,
        transforms=ToTensor()
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Determine number of classes from train_json
    with open(args.train_json) as f:
        train_data = json.load(f)
    num_classes = len(train_data['categories']) + 1  # including background
    model = load_model(args.checkpoint, num_classes, device)

    all_targets = []
    all_preds = []

    # Iterate over validation set
    for images, targets in val_loader:
        image = images[0].to(device)
        gt_boxes = targets[0]['boxes'].numpy()
        gt_labels = targets[0]['labels'].numpy()

        with torch.no_grad():
            outputs = model([image])
        pred = outputs[0]
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy() - 1  # zero-indexed

        # For each GT box, match to best pred by IoU
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            if len(pred_boxes) == 0:
                continue
            ious = box_iou_np(gt_box, pred_boxes)
            idx = np.argmax(ious)
            if ious[idx] >= args.iou_thresh:
                all_targets.append(gt_label)
                all_preds.append(pred_labels[idx])

    # Compute confusion matrix
    labels = list(range(len(train_data['categories'])))
    cm = confusion_matrix(all_targets, all_preds, labels=labels)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label', ylabel='True label',
           title='Confusion Matrix for Validation Set')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Annotate counts
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    # Save the confusion matrix figure
    output_dir = './confusion_matrix'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate checkpoint and plot confusion matrix')
    parser.add_argument('--data_dir', default='nycu-hw2-data', help='dataset root')
    parser.add_argument('--val_json', default='nycu-hw2-data/valid.json', help='path to validation JSON')
    parser.add_argument('--train_json', default='nycu-hw2-data/train.json', help='path to train JSON')
    parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for matching')
    args = parser.parse_args()
    main(args)
