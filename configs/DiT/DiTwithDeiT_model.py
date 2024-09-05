import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed
import timm

class DiTwithDeiT(nn.Module):
    def __init__(
            self,
            deit_model='deit_base_patch16_224',  # 사용할 DeiT 모델 선택
            img_size=32,  # input_size를 img_size로 변경
            patch_size=4,
            in_channels=4,
            hidden_size=768,
            num_classes=1000,
            depth=12,
            num_heads=12
        ):
        super().__init__()

        # TIMM 라이브러리를 사용하여 DeiT 모델 불러오기
        self.deit_backbone = timm.create_model(deit_model, pretrained=True)

        # DeiT 모델의 임베딩 레이어를 덮어쓰기 위해 PatchEmbed 사용
        self.x_embedder = PatchEmbed(img_size=(img_size, img_size), patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)

        # 기존 DiT 구조에 필요한 임베딩 레이어들
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.y_embedder = LabelEmbedder(num_classes=num_classes, hidden_size=hidden_size, dropout_prob=0.1)

        # 최종 레이어 설정
        self.final_layer = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels)

    def forward(self, x, t, y):
        # 임베딩 및 timestep 임베딩
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)

        # DeiT 백본을 통해 데이터 처리
        deit_output = self.deit_backbone.forward_features(x + t + y)

        # 최종 레이어 처리
        output = self.final_layer(deit_output, t + y)
        return output



#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Final Layer                                   #
#################################################################################

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

#################################################################################
#                                   DiTwithDeiT Configs                                 #
#################################################################################

def DiTwithDeiT_XL_2(**kwargs):
    return DiTwithDeiT(hidden_size=1152, patch_size=2, depth=28, num_heads=16, **kwargs)

def DiTwithDeiT_XL_4(**kwargs):
    return DiTwithDeiT(hidden_size=1152, patch_size=4, depth=28, num_heads=16, **kwargs)

def DiTwithDeiT_XL_8(**kwargs):
    return DiTwithDeiT(hidden_size=1152, patch_size=8, depth=28, num_heads=16, **kwargs)

def DiTwithDeiT_L_2(**kwargs):
    return DiTwithDeiT(hidden_size=1024, patch_size=2, depth=24, num_heads=16, **kwargs)

def DiTwithDeiT_L_4(**kwargs):
    return DiTwithDeiT(hidden_size=1024, patch_size=4, depth=24, num_heads=16, **kwargs)

def DiTwithDeiT_L_8(**kwargs):
    return DiTwithDeiT(hidden_size=1024, patch_size=8, depth=24, num_heads=16, **kwargs)

def DiTwithDeiT_B_2(**kwargs):
    return DiTwithDeiT(hidden_size=768, patch_size=2, depth=12, num_heads=12, **kwargs)

def DiTwithDeiT_B_4(**kwargs):
    return DiTwithDeiT(hidden_size=768, patch_size=4, depth=12, num_heads=12, **kwargs)

def DiTwithDeiT_B_8(**kwargs):
    return DiTwithDeiT(hidden_size=768, patch_size=8, depth=12, num_heads=12, **kwargs)

def DiTwithDeiT_S_2(**kwargs):
    return DiTwithDeiT(hidden_size=384, patch_size=2, depth=12, num_heads=6, **kwargs)

def DiTwithDeiT_S_4(**kwargs):
    return DiTwithDeiT(hidden_size=384, patch_size=4, depth=12, num_heads=6, **kwargs)

def DiTwithDeiT_S_8(**kwargs):
    return DiTwithDeiT(hidden_size=384, patch_size=8, depth=12, num_heads=6, **kwargs)



DiTwithDeiT_models = {
    'DiTwithDeiT-XL/2': DiTwithDeiT_XL_2,  'DiTwithDeiT-XL/4': DiTwithDeiT_XL_4,  'DiTwithDeiT-XL/8': DiTwithDeiT_XL_8,
    'DiTwithDeiT-L/2':  DiTwithDeiT_L_2,   'DiTwithDeiT-L/4':  DiTwithDeiT_L_4,   'DiTwithDeiT-L/8':  DiTwithDeiT_L_8,
    'DiTwithDeiT-B/2':  DiTwithDeiT_B_2,   'DiTwithDeiT-B/4':  DiTwithDeiT_B_4,   'DiTwithDeiT-B/8':  DiTwithDeiT_B_8,
    'DiTwithDeiT-S/2':  DiTwithDeiT_S_2,   'DiTwithDeiT-S/4':  DiTwithDeiT_S_4,   'DiTwithDeiT-S/8':  DiTwithDeiT_S_8,
}
