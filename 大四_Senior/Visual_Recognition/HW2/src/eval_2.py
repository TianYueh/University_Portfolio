import os
import json
import argparse
import csv
import math


def get_all_image_ids(test_dir):
    # 從 test 資料夾讀取所有檔案 (假設檔案名稱均為數字，例如 1.png, 2.jpg)
    img_ids = []
    for f in os.listdir(test_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_id = int(os.path.splitext(f)[0])
                img_ids.append(img_id)
            except ValueError:
                continue
    return sorted(img_ids)


def iou(boxA, boxB):
    # box format: [x_min, y_min, width, height]
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    x_left = max(xA1, xB1)
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = wA * hA
    boxB_area = wB * hB
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def main(args):
    # 讀取 Task 1 產生的檢測結果 pred.json
    pred_json_path = os.path.join(args.input_dir, 'pred.json')
    with open(pred_json_path) as f:
        detections = json.load(f)
    # 讀取 train.json 取得 category mapping
    with open(args.train_json) as f:
        train_data = json.load(f)
    cat_map = {c['id']: str(c['name']) for c in train_data['categories']}

    # 收集檢測結果
    img_results = {}
    for det in detections:
        img_results.setdefault(det['image_id'], []).append(det)

    all_img_ids = get_all_image_ids(os.path.join(args.data_dir, 'test'))
    classifications = []

    for img_id in all_img_ids:
        dets = img_results.get(img_id, [])
        if not dets:
            pred_label = '-1'
        else:
            # 排序
            dets_sorted = sorted(dets, key=lambda d: d['bbox'][0])
            # 重複抑制
            filtered = []
            for d in dets_sorted:
                keep = True
                for f in filtered:
                    if f['category_id'] == d['category_id'] and iou(f['bbox'], d['bbox']) > args.iou_thresh:
                        keep = False
                        break
                if keep:
                    filtered.append(d)
            # 離群點移除
            if len(filtered) > 2:
                centers = [(d, (d['bbox'][0] + d['bbox'][2]/2, d['bbox'][1] + d['bbox'][3]/2)) for d in filtered]
                kept = []
                for d, (cx, cy) in centers:
                    dists = [math.hypot(cx - ocx, cy - ocy) for od, (ocx, ocy) in centers if od is not d]
                    if min(dists) <= args.dist_thresh:
                        kept.append(d)
                if kept:
                    filtered = kept
            elif len(filtered) == 2:
                d1, d2 = filtered
                cx1, cy1 = d1['bbox'][0] + d1['bbox'][2]/2, d1['bbox'][1] + d1['bbox'][3]/2
                cx2, cy2 = d2['bbox'][0] + d2['bbox'][2]/2, d2['bbox'][1] + d2['bbox'][3]/2
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                if dist > args.dist_thresh:
                    filtered = [d1] if d1['score'] >= d2['score'] else [d2]
            # 組合結果
            digits = [cat_map.get(d['category_id'], '') for d in filtered]
            pred_label = ''.join(digits) if digits else '-1'
        classifications.append((img_id, pred_label))

    # 輸出 CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'pred.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'pred_label'])
        for img_id, label in classifications:
            writer.writerow([img_id, label])
    print(f"Saved classification results to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Task2: Generate digit recognition results from Task1 output with suppression and outlier removal")
    parser.add_argument('--data_dir', default='nycu-hw2-data', help='dataset root directory')
    parser.add_argument('--input_dir', default='results', help='directory containing pred.json')
    parser.add_argument('--train_json', default='nycu-hw2-data/train.json', help='path to train json')
    parser.add_argument('--output_dir', default='results', help='directory to save pred.csv')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for duplicate suppression')
    parser.add_argument('--dist_thresh', type=float, default=50.0, help='Distance threshold to filter outliers')
    args = parser.parse_args()
    main(args)
