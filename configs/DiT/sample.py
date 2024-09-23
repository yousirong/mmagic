# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from a pre-trained DiT or DiT_Lightweight.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from models_lightweight import DiT_Lightweight_models  # DiT_Lightweight_models import 추가
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8

    # DiT_Lightweight 모델과 DiT 모델을 구분하여 불러오는 로직 추가
    if args.model in DiT_models:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        ).to(device)
    elif args.model in DiT_Lightweight_models:
        model = DiT_Lightweight_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        ).to(device)
    else:
        raise ValueError(f"Model {args.model} not found in available models.")

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"{args.model.replace('/', '-')}-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample1.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DiT_Lightweight 모델도 선택할 수 있도록 model choices를 확장
    all_model_choices = list(DiT_models.keys()) + list(DiT_Lightweight_models.keys())
    parser.add_argument("--model", type=str, choices=all_model_choices, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT or DiT_Lightweight checkpoint (default: auto-download a pre-trained model).")
    args = parser.parse_args()
    main(args)

"""
python sample.py \
    --model DiT_Lightweight-S/2 \
    --image-size 256 \
    --num-classes 1000 \
    --cfg-scale 4.0 \
    --num-sampling-steps 250 \
    --seed 42 \
    --ckpt /home/juneyonglee/mydata/results/000-DiT_Lightweight-S-2/checkpoints/0800000.pt \
    --vae ema

python sample.py \
    --model DiT_Lightweight-S/2 \
    --image-size 256 \
    --seed 42 \
    --ckpt /home/juneyonglee/mydata/results/000-DiT_Lightweight-S-2/checkpoints/0800000.pt \


    python sample.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --num-classes 1000 \
    --cfg-scale 4.0 \
    --num-sampling-steps 250 \
    --seed 42 \
    --vae ema


"""