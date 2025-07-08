# test.py
import os
import json
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from ddpm import DiffuserDDPM
from evaluator import evaluation_model
from objects import OBJECTS_MAP

def load_conditions(json_path):
    """
    读取 JSON。
    - 如果是字典格式 {fname: [labels]...}，返回 dict。
    - 如果是列表格式 [[labels], ...]，返回 list。
    """
    with open(json_path) as f:
        return json.load(f)

def onehot_from_conditions(conds):
    """
    将条件转换为 one-hot 编码张量。
    支持 dict {fname: [labels], ...} 和 list [[labels], ...]。
    返回形状 [N, num_classes] 的 Tensor y。
    """
    if isinstance(conds, dict):
        cond_list = list(conds.values())
    elif isinstance(conds, list):
        cond_list = conds
    else:
        raise ValueError("Unsupported JSON format for conditions")

    num = len(cond_list)
    num_classes = len(OBJECTS_MAP)
    y = torch.zeros(num, num_classes, dtype=torch.float)
    for i, labels in enumerate(cond_list):
        for obj_name in labels:
            idx = OBJECTS_MAP.get(obj_name)
            if idx is not None:
                y[i, idx] = 1
    return y

def test(json_path, ckpt, out_dir, mode, device):
    # 1. 加载条件和模型
    conds = load_conditions(json_path)
    y = onehot_from_conditions(conds).to(device)

    model = DiffuserDDPM().to(device)
    model.load_state_dict(torch.load(ckpt)['model'])
    model.eval()

    evaler = evaluation_model()
    evaler.resnet18 = evaler.resnet18.to(device)

    # 2. 生成样本
    with torch.no_grad():
        samples = model.sample(y, device)
        samples = (samples.clamp(-1,1) + 1) / 2  # 归一到 [0,1]

    # 3. 保存个别图像
    individual_dir = os.path.join(out_dir, mode)
    os.makedirs(individual_dir, exist_ok=True)
    for idx, img in enumerate(samples.cpu()):
        save_image(img, os.path.join(individual_dir, f"{idx}.png"))

    # 4. 保存整体图像网格
    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid(samples.cpu(), nrow=8)
    grid_path = os.path.join(out_dir, f"{mode}_samples.png")
    save_image(grid, grid_path)

    # 5. 评估分类准确率
    imgs_norm = samples * 2 - 1
    acc = evaler.eval(imgs_norm, y)
    print(f'{mode} set ({json_path}) accuracy: {acc*100:.2f}%')
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['test','new'], required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--json_dir', required=True)
    parser.add_argument('--out_dir', default='./test')
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'test':
        json_path = os.path.join(args.json_dir, 'test.json')
        mode_name = 'test'
    else:
        json_path = os.path.join(args.json_dir, 'new_test.json')
        mode_name = 'new_test'

    test(json_path, args.ckpt, args.out_dir, mode_name, device)
