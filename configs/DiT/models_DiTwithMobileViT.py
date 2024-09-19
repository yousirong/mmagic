import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed
from mmpretrain.models.backbones import MobileViT
from models import DiTBlock

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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
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
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of the DiT model, which converts the final transformer output into image space.
    """
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


class MobileViT_DiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=2,
        in_channels=3,
        hidden_size=384,  # MobileViT 출력 채널 수와 맞춤
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # MobileViT 로드 및 채널 조정
        self.mobilevit = MobileViT(arch='small', in_channels=in_channels, out_indices=(3,))

        # MobileViT의 출력 채널 수를 hidden_size로 맞추기 위한 Conv2d 레이어 (128 -> 384로 변환)
        self.conv_to_transformer = nn.Conv2d(128, hidden_size, kernel_size=1)

        # DiT 관련 설정
        self.x_embedder = PatchEmbed(input_size, patch_size, hidden_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        # MobileViT로 특징 추출
        x = self.mobilevit(x)[0]  # MobileViT 출력
        print(f"MobileViT output shape: {x.shape}")  # MobileViT의 출력 확인
        x = self.conv_to_transformer(x)  # 채널 수 맞춤
        print(f"After conv_to_transformer shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)  # Transformer가 처리할 수 있게 변경

        # Positional embedding 추가
        x = x + self.pos_embed

        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y  # 시간 및 레이블 임베딩

        # Transformer blocks 통과
        for block in self.blocks:
            x = block(x, c)

        # 최종 레이어
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1], "Patch 크기 불일치"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))


#################################################################################
# DiT Configurations with MobileViT
#################################################################################

def MobileViT_DiT_S_2(**kwargs):
    return MobileViT_DiT(hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def MobileViT_DiT_S_4(**kwargs):
    return MobileViT_DiT(hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def MobileViT_DiT_S_8(**kwargs):
    return MobileViT_DiT(hidden_size=384, patch_size=8, num_heads=6, **kwargs)


MobileViT_DiT_configs = {
    'MobileViT-DiT-S/2': MobileViT_DiT_S_2,
    'MobileViT-DiT-S/4': MobileViT_DiT_S_4,
    'MobileViT-DiT-S/8': MobileViT_DiT_S_8,
}
