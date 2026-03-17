"""Async embedding DataLoader.

Wraps a raw image dataset and runs a model on a secondary device in a
background thread, yielding (embedding, label) batches without writing to disk.

Usage::

    from hycut.datavault import EmbeddingDataLoader, load_model, get_dataset

    model, preprocess = load_model("clipvitL14", device="cuda:1", models_dir="./models")
    dataset = get_dataset("cifar10", split="train", transform=preprocess, root_dir="./data")
    loader = EmbeddingDataLoader(dataset, model, device="cuda:1", batch_size=256)

    for embeddings, labels in loader:
        # embeddings: (B, D) tensor on CPU
        # labels: (B,) tensor on CPU
        ...
"""

from __future__ import annotations

import threading
from queue import Queue
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EmbeddingDataLoader:
    """DataLoader that embeds images on-the-fly using a model on a given device.

    Parameters
    ----------
    dataset:
        A PyTorch Dataset that returns (image_tensor, label).
    model:
        A callable (e.g. ``model.encode_image``) that maps a batch of images to
        embeddings.
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
        model: nn.Module | callable,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch: int = 2,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
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
        error_holder: list = []  # shared mutable to propagate exceptions

        def _producer():
            try:
                with torch.no_grad():
                    for batch in self._loader:
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
