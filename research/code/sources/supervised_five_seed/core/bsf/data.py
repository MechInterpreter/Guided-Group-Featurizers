"""Rabbit images -> DINOv3 patch activations.

`rabbit.npz` holds 300 RGB images of size 224x224 (key ``arr_0``, uint8). We run
them through DINOv3 (fp32) and keep the patch tokens. Centre by the ImageNet
per-position mean and scale so the mean squared norm of an activation is ``d``
(||x|| ~ sqrt(d)) -- the convention the featurizers expect (done in the notebook).
"""
import pathlib

import numpy as np
import torch

# DINOv3-vitb16 per-patch-position mean over ~25M ImageNet patches (196, 768).
# This is the positional main effect -- across diverse classes the content
# averages out, leaving the position/register artifact. Subtracting it removes
# position without removing content (unlike a single-class mean).
POS_MEAN_PATH = pathlib.Path(__file__).with_name('pos_mean.npy')
# (n_patches, d)
POS_MEAN = np.load(POS_MEAN_PATH).astype(np.float32)

# DINOv3 ViT-B/16: 224px / patch 16 -> 14x14 = 196 patch tokens, d = 768.
# The first 5 tokens are CLS + 4 register tokens; we drop them.
DINO_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
N_REGISTER_TOKENS = 5


def load_rabbit_images(npz_path):
    """Load the rabbit images as a (N, H, W, 3) uint8 array."""
    return np.load(npz_path)['arr_0']


@torch.no_grad()
def dino_activations(images, *, device=None, batch_size=64, model_id=DINO_ID):
    """Images -> (N_imgs, n_patches, d) fp32 DINOv3 patch activations.

    `images` is the (N, H, W, 3) uint8 array from `load_rabbit_images`.
    """
    from transformers import AutoImageProcessor, AutoModel
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    proc = AutoImageProcessor.from_pretrained(model_id)
    # fp32
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    out = []
    for s in range(0, len(images), batch_size):
        chunk = [im for im in images[s:s + batch_size]]
        batch = proc(images=chunk, return_tensors='pt').to(device)
        # (B, T, d)
        h = model(**batch).last_hidden_state
        out.append(h[:, N_REGISTER_TOKENS:].float().cpu())
    return torch.cat(out).numpy()


def patch_grid(n_patches):
    """Side length of the square patch grid (14 for DINOv3 ViT-B/16 @ 224)."""
    g = int(round(n_patches ** 0.5))
    assert g * g == n_patches, f'{n_patches} patches is not a square grid'
    return g
