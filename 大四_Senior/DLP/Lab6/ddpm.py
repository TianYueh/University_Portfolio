import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler


class DiffuserDDPM(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        time_embed_dim: int = 512,            # 缩小 embedding 维度
        sample_size: int = 64,
        beta_schedule: str = "squaredcos_cap_v2",
        guidance_scale: float = 2.0,
    ):
        super().__init__()
        self.guidance_scale = guidance_scale

        # Scheduler for noise
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )

        # UNet2DModel
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[128,128,256,256,512,512], 
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            up_block_types=[
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            class_embed_type="identity",
        )

        # 限制注意力切片（Attention Slicing），降低一次性显存占用
        #self.unet.enable_attention_slicing()
        # 开启梯度检查点，换算成更多前向但更少后向显存
        #self.unet.enable_gradient_checkpointing()

        # 条件 embedding
        self.class_embedding = nn.Linear(num_classes, time_embed_dim)

    def forward(self, x, timesteps, y):
        y_embed = self.class_embedding(y)
        return self.unet(x, timesteps, y_embed).sample

    def sample(self, y, device, num_inference_steps=1000):
        b = y.shape[0]
        x = torch.randn(b, 3, 64, 64, device=device)
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)

        y_zero = torch.zeros_like(y)
        for t in scheduler.timesteps:
            eps_uncond = self.unet(x, t, self.class_embedding(y_zero)).sample
            eps_cond   = self.unet(x, t, self.class_embedding(y)).sample
            eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            step = scheduler.step(eps, t, x)
            x = step.prev_sample

        return x

    def add_noise(self, x, noise, timesteps):
        return self.scheduler.add_noise(x, noise, timesteps)
