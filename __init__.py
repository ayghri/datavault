from .registry import (
    register,
    list_datasets,
    get_dataset,
    get_datasets,
    get_dataloaders,
    get_spec,
)
from .datasets import ArrayDataset
from .utils import get_default_transforms
from .models import list_models, load_model, extract
from .embedding_loader import EmbeddingDataLoader

from pathlib import Path
from typing import Optional
import numpy as np


def load_embeddings(
    dataset_name: str,
    model_name: str,
    repr_dir: str | Path,
    split: str = "train",
) -> ArrayDataset:
    """Load pre-computed embeddings as an ArrayDataset.

    Expects files at ``repr_dir/dataset_name/model_name/feats_{split}.npy``
    and ``repr_dir/dataset_name/model_name/y_{split}.npy``.
    """
    base = Path(repr_dir) / dataset_name / model_name
    feats_path = base / f"feats_{split}.npy"
    labels_path = base / f"y_{split}.npy"
    if not feats_path.exists():
        raise FileNotFoundError(f"Features not found: {feats_path}")
    labels = labels_path if labels_path.exists() else None
    return ArrayDataset(str(feats_path), str(labels) if labels else None)


__all__ = [
    # datasets
    "register",
    "list_datasets",
    "get_dataset",
    "get_datasets",
    "get_dataloaders",
    "get_spec",
    "ArrayDataset",
    "get_default_transforms",
    # models
    "list_models",
    "load_model",
    "extract",
    # embeddings
    "load_embeddings",
    "EmbeddingDataLoader",
]
