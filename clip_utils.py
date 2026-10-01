# clip_utils.py

import torch
import torch.nn.functional as F


def clip_preprocess(x, image_resolution=224):
    """
    x: torch.Tensor (B, 3, H, W) or (3, H, W)
    returns: (B, 3, 224, 224)
    """

    if x.dim() == 3:
        x = x.unsqueeze(0)

    # Resize
    x = F.interpolate(
        x,
        size=(image_resolution, image_resolution),
        mode="bicubic",
        align_corners=False
    )

    return x