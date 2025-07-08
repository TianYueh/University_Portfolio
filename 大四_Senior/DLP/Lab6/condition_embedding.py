import torch.nn as nn

class ConditionEmbedding(nn.Module):
    def __init__(self, label_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(label_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, y):
        return self.proj(y)
