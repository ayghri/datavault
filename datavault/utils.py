from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    return image.convert("RGB")


def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    return transforms.ToTensor()(x)


def get_default_transforms():
    return transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            _safe_to_tensor,
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )


def load_np_array(arr: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(arr, (str, Path)):
        return np.load(arr)
    return arr
