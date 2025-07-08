import os
import json
import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# ─── Utility functions ──────────────────────────────────────────────────────────

def read_tiff(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {filepath}")
    # if 3-channel, convert to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    # pycocotools returns bytes for counts
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

# ─── Dataset for test set ──────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, test_dir, info_list):
        self.test_dir = test_dir
        self.info = info_list

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        rec = self.info[idx]
        img_path = os.path.join(self.test_dir, rec['file_name'])
        img = read_tiff(img_path)
        # convert to tensor and 3-channel
        t = F.to_tensor(img)            # 1×H×W
        t = t.repeat(3, 1, 1)           # 3×H×W
        return t, rec

# ─── Model constructor ─────────────────────────────────────────────────────────

def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
    model = MaskRCNN(backbone, num_classes=num_classes)
    # match your training transform
    model.transform = GeneralizedRCNNTransform(
        min_size=512, max_size=512,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    return model

# ─── Paths and device ──────────────────────────────────────────────────────────

DATA_DIR   = './hw3-data-release'
TEST_DIR   = os.path.join(DATA_DIR, 'test_release')
TEST_JSON  = os.path.join(DATA_DIR, 'test_image_name_to_ids.json')
MODEL_PATH = 'best_model.pth'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ─── Load model ────────────────────────────────────────────────────────────────

model = get_model(num_classes=5)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ─── Read test metadata ────────────────────────────────────────────────────────

with open(TEST_JSON, 'r') as f:
    test_info = json.load(f)

test_dataset = TestDataset(TEST_DIR, test_info)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# ─── Run inference ──────────────────────────────────────────────────────────────

test_results = []
score_thresh = 0.55  

with torch.no_grad():
    for tensors, rec in test_loader:
        tensors = [t.to(device) for t in tensors]
        outputs = model(tensors)[0]

        # filter low-score detections
        keep = outputs['scores'] > score_thresh
        boxes  = outputs['boxes'][keep].cpu().numpy()
        scores = outputs['scores'][keep].cpu().numpy()
        labels = outputs['labels'][keep].cpu().numpy()
        masks  = outputs['masks'][keep].cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            bin_mask = mask[0] > 0.5
            if not bin_mask.any():
                continue

            rle = encode_mask(bin_mask)
            x1, y1, x2, y2 = box
            test_results.append({
                'image_id':   int(rec['id']),
                'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                'score':      float(score),
                'category_id': int(label),
                'segmentation': {
                    'size':   [int(rec['height']), int(rec['width'])],
                    'counts': rle['counts']
                }
            })

# ─── Save submission ───────────────────────────────────────────────────────────

with open('submission.json', 'w') as f:
    json.dump(test_results, f)

# ─── 印出通過門檻的 instance 數量 ───────────────────────────────────────────────
print(f'Inference complete. {len(test_results)} instances passed the score threshold of {score_thresh}.')