import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from ddpm import DiffuserDDPM
from data_loader import get_dataloader

def get_random_timesteps(batch_size, num_timesteps, device):
    return torch.randint(0, num_timesteps, (batch_size,), device=device).long()

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2')
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--log_dir', default='./logs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffuserDDPM(
        beta_schedule=args.beta_schedule,
        guidance_scale=args.guidance_scale,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()  # AMP

    mse = nn.MSELoss()
    loader = get_dataloader(args.train_json, args.image_dir, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            # classifier-free guidance drop
            y_input = torch.zeros_like(y) if torch.rand(1).item() < args.drop_prob else y

            # add noise
            noise = torch.randn_like(x)
            t = get_random_timesteps(x.size(0), model.scheduler.num_train_timesteps, device)
            x_noisy = model.add_noise(x, noise, t)

            # forward + loss under autocast
            with autocast():
                noise_pred = model(x_noisy, t, y_input)
                loss = mse(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            losses.append(loss.item())
            pbar.set_postfix_str(f"loss: {np.mean(losses):.4f}")

        lr_scheduler.step()
        writer.add_scalar('train/loss', np.mean(losses), epoch)
        writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

        ckpt_path = os.path.join(args.out_dir, f'ddpm_epoch{epoch}.pth')
        torch.save({'model': model.state_dict()}, ckpt_path)

    writer.close()

if __name__ == '__main__':
    train()
