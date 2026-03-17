"""Model registry for datavault.

Provides a unified way to load embedding models (CLIP variants, DINOv2, etc.)
and extract representations from datasets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import _convert_image_to_rgb, _safe_to_tensor


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    loader: Callable[..., Tuple[nn.Module, Optional[object]]]
    hub_name: str = ""


_MODELS: dict[str, ModelSpec] = {}


def register_model(name: str, *, hub_name: str = ""):
    """Decorator to register a model loader.

    The decorated function must have signature:
        fn(device, models_dir) -> (model_callable, preprocess_or_None)
    """

    def wrapper(fn):
        _MODELS[name] = ModelSpec(name=name, loader=fn, hub_name=hub_name)
        return fn

    return wrapper


def list_models() -> list[str]:
    return sorted(_MODELS.keys())


def load_model(
    name: str,
    device: torch.device | str = "cpu",
    models_dir: str = "./models",
) -> Tuple[nn.Module, Optional[object]]:
    """Load a model by name. Returns (model_callable, preprocess_or_None)."""
    if name not in _MODELS:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    if isinstance(device, str):
        device = torch.device(device)
    return _MODELS[name].loader(device, models_dir)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(
    dataloader: DataLoader,
    model: nn.Module | Callable,
    device: torch.device,
    save_dir: Optional[str | Path] = None,
    suffix: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a dataloader using a model.

    If save_dir is provided, saves feats_{suffix}.npy and y_{suffix}.npy.
    Returns (features, labels) as numpy arrays.
    """
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Extracting {suffix}"):
            x = x.to(device)
            with torch.autocast(device_type=device.type):
                features = model(x)
            all_features.append(features.cpu())
            all_labels.append(y.cpu())

    feats = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"feats_{suffix}.npy", feats)
        np.save(save_dir / f"y_{suffix}.npy", labels)

    return feats, labels


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------

# --- CLIP variants ---

_CLIP_VARIANTS = {
    "clipRN50": "RN50",
    "clipRN101": "RN101",
    "clipRN50x4": "RN50x4",
    "clipRN50x16": "RN50x16",
    "clipRN50x64": "RN50x64",
    "clipvitB32": "ViT-B/32",
    "clipvitB16": "ViT-B/16",
    "clipvitL14": "ViT-L/14",
}


def _make_clip_loader(clip_arch: str):
    """Factory that creates a loader function for a specific CLIP architecture."""

    def _load(device, models_dir):
        import clip

        ckpt_dir = Path(models_dir) / "clip"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model, preprocess = clip.load(
            clip_arch, device=device, download_root=str(ckpt_dir)
        )
        model.eval()
        encoder = model.encode_image

        # Patch preprocessing to handle edge cases
        preprocess.transforms[2] = _convert_image_to_rgb
        preprocess.transforms[3] = _safe_to_tensor

        return encoder, preprocess

    return _load


for _name, _arch in _CLIP_VARIANTS.items():
    _MODELS[_name] = ModelSpec(
        name=_name, loader=_make_clip_loader(_arch), hub_name=_arch
    )


# --- DINOv2 ---

@register_model("dinov2", hub_name="facebookresearch/dinov2")
def _dinov2(device, models_dir):
    ckpt_dir = Path(models_dir) / "dinov2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(ckpt_dir))

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").to(device)
    model.eval()
    return model, None  # DINOv2 uses default transforms
