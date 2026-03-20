"""Embedding utilities: loading pre-computed embeddings and on-the-fly inference.

- ``ArrayDataset``: wraps saved .npy feature/label files as a PyTorch Dataset.
- ``EmbeddingDataLoader``: runs a model on a secondary device in a background
  thread, yielding (embedding, label) batches without writing to disk.
"""


from pathlib import Path
from typing import Callable, Optional, Tuple
from queue import Queue
import threading

from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from .utils import load_np_array


class ArrayDataset(Dataset):
    """PyTorch Dataset backed by pre-computed numpy arrays (features + labels)."""

    def __init__(
        self,
        features_arr: np.ndarray | str | Path,
        labels_arr: Optional[np.ndarray | str | Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.feats = load_np_array(features_arr)
        if labels_arr is None:
            self.labels = None
        else:
            self.labels = load_np_array(labels_arr)
            assert self.labels.shape[0] == self.feats.shape[0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        feat = self.feats[index]
        if self.transform is not None:
            feat = self.transform(feat)

        if self.labels is None:
            return feat
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feat, label


class EmbeddingDataLoader:
    """DataLoader that embeds images on-the-fly using a model on a given device.

    Wraps a raw image dataset, runs the model in a background thread, and
    yields ``(embedding, label)`` batches via a prefetch queue.

    Parameters
    ----------
    dataset:
        A PyTorch Dataset returning (image_tensor, label) or a dict with
        ``"image"`` and ``"label"`` keys.
    model:
        A callable that maps a batch of images to embeddings.
    device:
        The device to run the model on (e.g. ``"cuda:1"``).
    batch_size:
        Batch size for the underlying DataLoader.
    num_workers:
        Number of data-loading workers.
    prefetch:
        How many batches to embed ahead of time in the background.
    pin_memory:
        Whether to pin memory in the underlying DataLoader.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module | Callable,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch: int = 2,
        pin_memory: bool = True,
    ):
        self.model = model
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.prefetch = prefetch

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        queue: Queue = Queue(maxsize=self.prefetch)
        error_holder: list = []

        def _producer():
            try:
                with torch.no_grad():
                    for batch in self._loader:
                        if isinstance(batch, dict):
                            images, labels = batch["image"], batch["label"]
                        else:
                            images, labels = batch
                        images = images.to(self.device, non_blocking=True)
                        with torch.autocast(device_type=self.device.type):
                            embeddings = self.model(images)
                        queue.put((embeddings.cpu(), labels))
            except Exception as e:
                error_holder.append(e)
            finally:
                queue.put(None)  # sentinel

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is None:
                break
            yield item

        thread.join()
        if error_holder:
            raise error_holder[0]


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
