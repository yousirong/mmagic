import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.size()
        H = self.num_heads

        q = self.q_proj(x).reshape(B, N, H, self.head_dim)
        k = self.k_proj(x).reshape(B, N, H, self.head_dim)
        v = self.v_proj(x).reshape(B, N, H, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, H, N, head_dim)

        q = q * self.scale

        attn = torch.bmm(q.flatten(0, 1), k.flatten(0, 1).transpose(1, 2))
        attn = torch.nn.functional.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.flatten(0, 1))
        out = out.reshape(B, H, N, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        out = self.out_proj(out)
        return out
