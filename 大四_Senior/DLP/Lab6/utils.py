import torch
import torch.nn as nn
import math

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        sinusoid = torch.cat([
            torch.sin(t.unsqueeze(-1) * self.inv_freq),
            torch.cos(t.unsqueeze(-1) * self.inv_freq)
        ], dim=-1)
        return sinusoid
