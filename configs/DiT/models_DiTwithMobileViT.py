import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed
from mmpretrain.models.backbones import MobileViT
from mmagic.registry import MODELS

def modulate(x, shift, scale):
    # Reshape shift and scale to match x's shape if needed
    shift = shift.view(x.size(0), -1, 1, 1)  # Ensure shift has shape [batch_size, channels, 1, 1]
    scale = scale.view(x.size(0), -1, 1, 1)  # Ensure scale has shape [batch_size, channels, 1, 1]
    return x * (1 + scale) + shift

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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

#################################################################################
#                                 Core MobileViT DiT Model                      #
#################################################################################

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)  # Updated to match MobileViT output channels
        self.linear = nn.Linear(128, patch_size * patch_size * out_channels, bias=True)  # Updated to 128 channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * 128, bias=True)  # Update to match 128-channel output
        )

    def forward(self, x, c):
        # Flatten the spatial dimensions to get a 2D tensor
        x = x.view(x.size(0), -1)  # From [64, 128, 1, 1] to [64, 128]
        
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Update the MobileViT_DiT class initialization
class MobileViT_DiT(nn.Module):
    def __init__(
        self,
        input_size=256,  # Updated to 256 to match your ImageNet dataset
        patch_size=2,
        in_channels=3,  # Updated to 3 for RGB inputs
        hidden_size=384,   # Use 384 to match the MobileViT's expected input
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

        # Ensure PatchEmbed outputs the correct number of channels
        self.x_embedder = PatchEmbed(input_size, patch_size, 3, hidden_size)  # Outputs 384 channels now

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Add a convolution to ensure input to MobileViT has 384 channels
        self.pre_mobilevit_conv = nn.Conv2d(384, hidden_size, kernel_size=1)  # Channel conversion to 384

        # MobileViT expects input channels to match PatchEmbed's output
        self.mobilevit = MobileViT(arch='small', in_channels=hidden_size, out_indices=(3,))  # MobileViT receives 384 channels

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

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size

        print(f"Initial x.shape: {x.shape}")
        
        if x.shape[2] == 1 and x.shape[3] == 1:
            num_patches = x.shape[1]
            h = w = int(math.sqrt(num_patches))
            print(f"Handling 1x1 spatial dimensions. Computed h={h}, w={w}, num_patches={num_patches}")

            assert h * w == num_patches, f"Expected h * w to match x.shape[1], but got h={h}, w={w}, and x.shape[1]={num_patches}"

            x = x.view(x.shape[0], c, h, w)
            x = x.repeat(1, 1, p, p)
        else:
            num_patches = x.shape[1]
            h = w = int(math.sqrt(num_patches))
            print(f"Handling general case. Computed h={h}, w={w}, num_patches={num_patches}")
            
            assert h * w == num_patches, f"Expected h * w to match x.shape[1], but got h={h}, w={w}, and x.shape[1]={num_patches}"
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], c, h * p, w * p))

        print(f"Final x.shape: {x.shape}")
        return x




    def forward(self, x, t, y):
        print("Input shape before slice:", x.shape)  # Check initial input shape
        if x.shape[1] == 4:  # If input has 4 channels, keep only the first 3 (RGB)
            x = x[:, :3, :, :]
        print("Input shape after slice:", x.shape)  # Check after slicing
        x = self.x_embedder(x)
        print("After PatchEmbed:", x.shape)  # Check after embedding
        
        # Reshape the output of PatchEmbed to have 4 dimensions
        # Reshape the output of PatchEmbed to have 4 dimensions
        batch_size, num_patches, hidden_size = x.shape
        height = width = int(math.sqrt(num_patches))  # Assuming square patches
        x = x.reshape(batch_size, hidden_size, height, width)  # Use reshape instead of view
        print(f"After reshaping PatchEmbed: {x.shape}")
        
        # Add pre-MobileViT convolution
        x = self.pre_mobilevit_conv(x)
        
        # Check the shape of x before feeding to MobileViT
        print(f"Shape before MobileViT: {x.shape}")
        
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        print("Combined shape (t + y):", c.shape)  # Check combined embedding shape
        
        # Unpack the first element from the tuple returned by MobileViT
        x = self.mobilevit(x)[0]  # Extract the first element which is the output tensor
        print("After MobileViT:", x.shape)  # Check after MobileViT
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        print("Final output shape:", x.shape)  # Final output shape
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                                   DiT Configs                                  #
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
