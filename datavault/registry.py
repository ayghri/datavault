"""Dataset registry for datavault.

Each dataset is registered as a callable that returns (train_dataset, val_dataset).
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from .utils import get_default_transforms


@dataclass
class DatasetSpec:
    """Metadata for a registered dataset."""

    name: str
    loader: Callable[..., tuple]  # (transform, data_path) -> (train, val)
    notes: str = ""


_REGISTRY: dict[str, DatasetSpec] = {}


def register(name: str, *, notes: str = ""):
    """Decorator to register a dataset loader function.

    The decorated function must have signature:
        fn(transform, data_path: str) -> (train_dataset, val_dataset)
    """

    def wrapper(fn):
        _REGISTRY[name] = DatasetSpec(name=name, loader=fn, notes=notes)
        return fn

    return wrapper


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        available = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _REGISTRY[name]


def get_dataset(
    name: str,
    split: str = "train",
    transform: Optional[transforms.Compose] = None,
    root_dir: str = "./data",
    **kwargs,
) -> torch.utils.data.Dataset:
    """Load a single split of a dataset by name.

    Extra kwargs (e.g. ``streaming=True``) are forwarded to the loader.
    """
    if transform is None:
        transform = get_default_transforms()
    spec = get_spec(name)
    data_path = os.path.join(root_dir, "datasets")
    train_ds, val_ds = spec.loader(transform, data_path, **kwargs)
    if split == "train":
        return train_ds
    elif split in ("val", "test"):
        return val_ds
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")


def get_datasets(
    name: str,
    transform: Optional[transforms.Compose] = None,
    root_dir: str = "./data",
    **kwargs,
) -> tuple:
    """Load both (train, val) splits of a dataset by name.

    Extra kwargs (e.g. ``streaming=True``) are forwarded to the loader.
    """
    if transform is None:
        transform = get_default_transforms()
    spec = get_spec(name)
    data_path = os.path.join(root_dir, "datasets")
    return spec.loader(transform, data_path, **kwargs)


def get_dataloaders(
    name: str,
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 64,
    root_dir: str = "./data",
    num_workers: int = 10,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Load both splits and wrap them in DataLoaders.

    Extra kwargs (e.g. ``streaming=True``) are forwarded to the dataset loader.
    """
    train_ds, val_ds = get_datasets(name, transform, root_dir, **kwargs)
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return trainloader, valloader


# ---------------------------------------------------------------------------
# Custom dataset classes needed by some loaders
# ---------------------------------------------------------------------------


class MyFER2013(dsets.FER2013):
    _RESOURCES = {
        "train": ("train_split.csv", "aa1bdf3e64bc6697783ce586283a2b74"),
        "test": ("test_split.csv", "8576e0f5a806d7b337d6eeda66d71dc0"),
    }


class MySUN397(dsets.SUN397):
    def __init__(
        self, root, transform, target_transform=None, partition=1, split="train"
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        self.partition = partition
        self.split = split
        self.filter()

    def filter(self):
        split_str = (
            f"Training_{self.partition:02d}.txt"
            if self.split == "train"
            else f"Testing_{self.partition:02d}.txt"
        )
        with open(self._data_dir / split_str) as f:
            self._image_files = f.read().splitlines()
            self._image_files = [
                self._data_dir / elem[1:] for elem in self._image_files
            ]

        self._labels = [
            self.class_to_idx[
                "/".join(path.relative_to(self._data_dir).parts[1:-1])
            ]
            for path in self._image_files
        ]


# ---------------------------------------------------------------------------
# Registered dataset loaders
# ---------------------------------------------------------------------------


@register("food101")
def _food101(transform, data_path):
    return (
        dsets.Food101(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.Food101(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register("cifar10")
def _cifar10(transform, data_path):
    return (
        dsets.CIFAR10(
            root=data_path, train=True, transform=transform, download=True
        ),
        dsets.CIFAR10(
            root=data_path, train=False, transform=transform, download=True
        ),
    )


@register("cifar100")
def _cifar100(transform, data_path):
    return (
        dsets.CIFAR100(
            root=data_path, train=True, transform=transform, download=True
        ),
        dsets.CIFAR100(
            root=data_path, train=False, transform=transform, download=True
        ),
    )


@register(
    "birdsnap",
    notes="Manually download wget https://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz "
    "and run python get_birdsnap.py (Requires python2), then run prepare_birdsnap.py",
)
def _birdsnap(transform, data_path):
    return (
        dsets.ImageFolder(
            root=os.path.join(data_path, "birdsnap/train"), transform=transform
        ),
        dsets.ImageFolder(
            root=os.path.join(data_path, "birdsnap/test"), transform=transform
        ),
    )


@register(
    "sun397",
    notes="Manually download SUN397.tar.gz and Partitions.zip, "
    "move Training_0*/Testing_0* splits to SUN397 dir",
)
def _sun397(transform, data_path):
    return (
        MySUN397(
            root=data_path, partition=1, split="train", transform=transform
        ),
        MySUN397(
            root=data_path, partition=1, split="test", transform=transform
        ),
    )


@register(
    "cars",
    notes="May require manual download from Kaggle. "
    "See https://github.com/pytorch/vision/issues/7545",
)
def _cars(transform, data_path):
    return (
        dsets.StanfordCars(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.StanfordCars(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register("aircraft")
def _aircraft(transform, data_path):
    return (
        dsets.FGVCAircraft(
            root=data_path, split="trainval", transform=transform, download=True
        ),
        dsets.FGVCAircraft(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register("dtd")
def _dtd(transform, data_path):
    train = ConcatDataset(
        (
            dsets.DTD(
                root=data_path,
                split="train",
                transform=transform,
                download=True,
            ),
            dsets.DTD(
                root=data_path, split="val", transform=transform, download=True
            ),
        )
    )
    val = dsets.DTD(
        root=data_path, split="test", transform=transform, download=True
    )
    return train, val


@register("pets")
def _pets(transform, data_path):
    return (
        dsets.OxfordIIITPet(
            root=data_path, split="trainval", transform=transform, download=True
        ),
        dsets.OxfordIIITPet(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register(
    "caltech101",
    notes="Requires pip install gdown for torchvision auto-download.",
)
def _caltech101(transform, data_path):
    full = dsets.Caltech101(root=data_path, transform=transform, download=True)
    targets = np.array(full.y)
    train_idx = []
    for t in np.unique(targets):
        np.random.seed(42)
        train_idx.extend(
            np.random.choice(np.where(targets == t)[0], size=30, replace=False)
        )
    val_idx = list(set(range(len(targets))) - set(train_idx))
    return Subset(full, train_idx), Subset(full, val_idx)


@register("flowers")
def _flowers(transform, data_path):
    train = ConcatDataset(
        (
            dsets.Flowers102(
                root=data_path,
                split="train",
                transform=transform,
                download=True,
            ),
            dsets.Flowers102(
                root=data_path, split="val", transform=transform, download=True
            ),
        )
    )
    val = dsets.Flowers102(
        root=data_path, split="test", transform=transform, download=True
    )
    return train, val


@register("mnist")
def _mnist(transform, data_path):
    return (
        dsets.MNIST(
            root=data_path, train=True, transform=transform, download=True
        ),
        dsets.MNIST(
            root=data_path, train=False, transform=transform, download=True
        ),
    )


@register("fashionmnist")
def _fashionmnist(transform, data_path):
    return (
        dsets.FashionMNIST(
            root=data_path, train=True, transform=transform, download=True
        ),
        dsets.FashionMNIST(
            root=data_path, train=False, transform=transform, download=True
        ),
    )


@register("imagenette")
def _imagenette(transform, data_path):
    return (
        dsets.Imagenette(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.Imagenette(
            root=data_path, split="val", transform=transform, download=True
        ),
    )


@register(
    "fer2013",
    notes="Download from Kaggle, split fer2013.csv to train/test, "
    "update md5sum in MyFER2013",
)
def _fer2013(transform, data_path):
    return (
        MyFER2013(root=data_path, split="train", transform=transform),
        MyFER2013(root=data_path, split="test", transform=transform),
    )


@register("stl10")
def _stl10(transform, data_path):
    return (
        dsets.STL10(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.STL10(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register(
    "eurosat",
    notes="Manually download EuroSAT.zip and unzip to ./data/datasets/eurosat",
)
def _eurosat(transform, data_path):
    full = dsets.EuroSAT(root=data_path, transform=transform, download=True)
    targets = np.array(full.targets)
    train_idx, val_idx = [], []
    for t in np.unique(targets):
        np.random.seed(42)
        subset = np.random.choice(
            np.where(targets == t)[0], size=1500, replace=False
        )
        train_idx.extend(subset[:1000])
        val_idx.extend(subset[1000:])
    return Subset(full, train_idx), Subset(full, val_idx)


@register("resisc45", notes="Uses HuggingFace timm/resisc45.")
def _resisc45(transform, data_path, streaming=False):
    import datasets as hf_datasets

    hf_id = "timm/resisc45"

    def _apply(batch):
        batch["image"] = [
            transform(img.convert("RGB")) for img in batch["image"]
        ]
        return batch

    def _apply_single(example):
        example["image"] = transform(example["image"].convert("RGB"))
        return example

    train = hf_datasets.load_dataset(
        hf_id,
        split="train",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )
    val = hf_datasets.load_dataset(
        hf_id,
        split="test",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )

    if streaming:
        train = train.map(_apply_single)
        val = val.map(_apply_single)
    else:
        train.set_transform(_apply)
        val.set_transform(_apply)

    return train, val


@register("gtsrb")
def _gtsrb(transform, data_path):
    return (
        dsets.GTSRB(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.GTSRB(
            root=data_path, split="test", transform=transform, download=True
        ),
    )


@register(
    "kitti",
    notes="Download image_2 and label_2 zips from avg-kitti, "
    "then run: python scripts/prepare_kitti.py (from turtle/dataset_preparation). "
    "Labels are vehicle distance bins (4 classes).",
)
def _kitti(transform, data_path):
    return (
        dsets.ImageFolder(
            root=os.path.join(data_path, "Kitti/train"), transform=transform
        ),
        dsets.ImageFolder(
            root=os.path.join(data_path, "Kitti/val"), transform=transform
        ),
    )


@register("country211")
def _country211(transform, data_path):
    train = ConcatDataset(
        (
            dsets.Country211(
                root=data_path,
                split="train",
                transform=transform,
                download=True,
            ),
            dsets.Country211(
                root=data_path,
                split="valid",
                transform=transform,
                download=True,
            ),
        )
    )
    val = dsets.Country211(
        root=data_path, split="test", transform=transform, download=True
    )
    return train, val


@register("pcam")
def _pcam(transform, data_path):
    train = ConcatDataset(
        (
            dsets.PCAM(
                root=data_path,
                split="train",
                transform=transform,
                download=True,
            ),
            dsets.PCAM(
                root=data_path, split="val", transform=transform, download=True
            ),
        )
    )
    val = dsets.PCAM(
        root=data_path, split="test", transform=transform, download=True
    )
    return train, val


@register(
    "ucf101",
    notes="Requires pip install av pyunpack patool. "
    "Run prepare_ucf101.py --download",
)
def _ucf101(transform, data_path):
    return (
        dsets.ImageFolder(
            root=os.path.join(data_path, "ucf101/train"), transform=transform
        ),
        dsets.ImageFolder(
            root=os.path.join(data_path, "ucf101/val"), transform=transform
        ),
    )


@register(
    "kinetics700",
    notes="Requires pip install av. Run prepare_k700.py --download. "
    "May take up to a day and a half.",
)
def _kinetics700(transform, data_path):
    return (
        dsets.ImageFolder(
            root=os.path.join(data_path, "k700/train_images"),
            transform=transform,
        ),
        dsets.ImageFolder(
            root=os.path.join(data_path, "k700/val_images"), transform=transform
        ),
    )


@register("clevr")
def _clevr(transform, data_path):
    return (
        dsets.CLEVRClassification(
            root=data_path, split="train", transform=transform, download=True
        ),
        dsets.CLEVRClassification(
            root=data_path, split="val", transform=transform, download=True
        ),
    )


@register(
    "hatefulmemes",
    notes="Manually download from Kaggle and unzip, then run prepare_memes.py",
)
def _hatefulmemes(transform, data_path):
    return (
        dsets.ImageFolder(
            root=os.path.join(data_path, "hatefulmemes/train"),
            transform=transform,
        ),
        dsets.ImageFolder(
            root=os.path.join(data_path, "hatefulmemes/test"),
            transform=transform,
        ),
    )


@register("sst")
def _sst(transform, data_path):
    train = ConcatDataset(
        (
            dsets.RenderedSST2(
                root=data_path,
                split="train",
                transform=transform,
                download=True,
            ),
            dsets.RenderedSST2(
                root=data_path, split="val", transform=transform, download=True
            ),
        )
    )
    val = dsets.RenderedSST2(
        root=data_path, split="test", transform=transform, download=True
    )
    return train, val


@register("cub", notes="Uses HuggingFace bentrevett/cub-200-2011.")
def _cub(transform, data_path, streaming=False):
    import datasets as hf_datasets

    hf_id = "bentrevett/cub-200-2011"

    def _apply(batch):
        batch["image"] = [
            transform(img.convert("RGB")) for img in batch["image"]
        ]
        return batch

    def _apply_single(example):
        example["image"] = transform(example["image"].convert("RGB"))
        return example

    train = hf_datasets.load_dataset(
        hf_id,
        split="train",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )
    val = hf_datasets.load_dataset(
        hf_id,
        split="test",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )

    if streaming:
        train = train.map(_apply_single)
        val = val.map(_apply_single)
    else:
        train.set_transform(_apply)
        val.set_transform(_apply)

    return train, val


@register(
    "imagenet",
    notes="Uses HuggingFace ILSVRC/imagenet-1k (gated — requires huggingface-cli login). "
    "Supports streaming=True.",
)
def _imagenet(transform, data_path, streaming=False):
    import datasets as hf_datasets

    hf_id = "ILSVRC/imagenet-1k"

    def _apply(batch):
        batch["image"] = [
            transform(img.convert("RGB")) for img in batch["image"]
        ]
        return batch

    def _apply_single(example):
        example["image"] = transform(example["image"].convert("RGB"))
        return example

    train = hf_datasets.load_dataset(
        hf_id,
        split="train",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )
    val = hf_datasets.load_dataset(
        hf_id,
        split="validation",
        streaming=streaming,
        cache_dir=None if streaming else data_path,
    )

    if streaming:
        train = train.map(_apply_single)
        val = val.map(_apply_single)
    else:
        train.set_transform(_apply)
        val.set_transform(_apply)

    return train, val
