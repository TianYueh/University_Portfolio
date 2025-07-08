# sample_denoise_grid.py
import torch
from torchvision.utils import make_grid, save_image
from ddpm import DiffuserDDPM
from objects import OBJECTS_MAP

def sample_with_denoise_grid(
    model, label_names, objects_map, device,
    num_inference_steps=1000, grid_steps=10,
    out_path="denoise_grid_cpu.png"
):
    # 构造 one-hot
    y = torch.zeros(1, len(objects_map), device=device)
    for name in label_names:
        y[0, objects_map[name]] = 1

    # 如果 diffusers 支持，可以尝试开启 slicing/offload
    try:
        model.unet.enable_attention_slicing()
        model.unet.enable_sequential_cpu_offload()
    except:
        pass

    # 全程在 CPU 上
    x = torch.randn(1, 3, 64, 64, device=device)
    scheduler = model.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    imgs = []
    # 只展示少量帧，减少中间缓存
    step_indices = torch.linspace(0, num_inference_steps-1, grid_steps).long().tolist()

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            eps_uncond = model.unet(x, t, model.class_embedding(torch.zeros_like(y))).sample
            eps_cond   = model.unet(x, t, model.class_embedding(y)).sample
            eps = eps_uncond + model.guidance_scale * (eps_cond - eps_uncond)
            x = scheduler.step(eps, t, x).prev_sample

        if i in step_indices:
            img = (x.clamp(-1,1) + 1) / 2
            imgs.append(img.cpu())  # 移到 CPU
            # torch.cuda.empty_cache()  # CPU 模式下可省略

    grid = make_grid(torch.cat(imgs, 0), nrow=len(imgs), pad_value=0.1)
    save_image(grid, out_path)
    print(f"Denoising grid saved to {out_path}")

if __name__ == "__main__":
    # 强制使用 CPU
    device = torch.device('cpu')
    model = DiffuserDDPM().to(device)
    model.load_state_dict(torch.load("checkpoints/ddpm_epoch200.pth", map_location='cpu')['model'])
    model.eval()

    sample_with_denoise_grid(
        model,
        label_names=["red sphere","cyan cylinder","cyan cube"],
        objects_map=OBJECTS_MAP,
        device=device,
        num_inference_steps=1000,   # 可再调小到 100
        grid_steps=10,              # 可再调小到 3
        out_path="denoise_process_cpu.png"
    )
