# datavault

Unified API for loading 28+ vision datasets and extracting/serving embeddings from any PyTorch model.

## Install

```bash
pip install -e .          # core (torchvision datasets)
pip install -e ".[hf]"   # + HuggingFace datasets (ImageNet streaming, etc.)
```

## Quick start

### Load a dataset

```python
from datavault import list_datasets, get_dataset, get_datasets, get_dataloaders

# See available datasets
print(list_datasets())

# Single split
train_ds = get_dataset("cifar10", split="train", root_dir="./data")

# Both splits
train_ds, val_ds = get_datasets("cifar10", root_dir="./data")

# Ready-made DataLoaders
train_loader, val_loader = get_dataloaders("cifar10", batch_size=128, root_dir="./data")
```

### ImageNet via HuggingFace (streaming)

No manual download needed — stream ImageNet-1k directly from HuggingFace:

```python
from datavault import get_datasets, get_dataloaders

# Full download (cached)
train_ds, val_ds = get_datasets("imagenet", root_dir="./data")

# Streaming — no disk usage, samples fetched on-the-fly
train_ds, val_ds = get_datasets("imagenet", streaming=True)

# get_dataloaders handles both dict-returning (HF) and tuple-returning
train_loader, val_loader = get_dataloaders("imagenet", batch_size=128, root_dir="./data")
```

Requires `pip install datasets` and `huggingface-cli login` (ImageNet-1k is a gated dataset).

### Extract and save embeddings

Bring any PyTorch model that accepts image tensors and returns feature vectors:

```python
import torch
from datavault import get_dataloaders, extract

# Your model — anything with a forward() that maps images -> embeddings
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
model.eval().to("cuda:0")

train_loader, val_loader = get_dataloaders("cifar10", batch_size=256, root_dir="./data")

# Extract and save to disk
extract(train_loader, model, device=torch.device("cuda:0"),
        save_dir="./representations/cifar10/dinov2", suffix="train")
extract(val_loader, model, device=torch.device("cuda:0"),
        save_dir="./representations/cifar10/dinov2", suffix="val")
```

### Load pre-computed embeddings

```python
from datavault import load_embeddings

# Loads feats_train.npy + y_train.npy as a Dataset
ds = load_embeddings("cifar10", "dinov2", repr_dir="./representations", split="train")
feat, label = ds[0]
```

### On-the-fly embeddings (async)

Run a model on a secondary device in a background thread — no disk I/O needed:

```python
from datavault import EmbeddingDataLoader, get_dataset

dataset = get_dataset("cifar10", split="train", transform=my_transform, root_dir="./data")

loader = EmbeddingDataLoader(
    dataset,
    model,                # any callable: images -> embeddings
    device="cuda:1",      # runs on a separate GPU
    batch_size=256,
    prefetch=2,           # batches buffered ahead
)

for embeddings, labels in loader:
    # embeddings: (B, D) CPU tensor
    # labels: (B,) CPU tensor
    ...
```

### Built-in CLIP/DINOv2 models (optional)

datavault ships with a small model registry for convenience. Requires `clip` (`pip install git+https://github.com/openai/CLIP.git`) for CLIP models.

```python
from datavault import list_models, load_model

print(list_models())
# ['clipRN50', 'clipRN101', ..., 'clipvitL14', 'dinov2']

model, preprocess = load_model("clipvitL14", device="cuda:0", models_dir="./models")
# preprocess is the CLIP transform; use it when loading datasets
# For DINOv2 preprocess is None — use get_default_transforms()
```

### Register a custom dataset

```python
from datavault import register

@register("my_dataset", notes="Description or setup instructions")
def _my_dataset(transform, data_path):
    train_ds = ...  # build your train dataset
    val_ds = ...    # build your val dataset
    return train_ds, val_ds
```

## Available datasets

Some datasets require manual download — check `get_spec("dataset_name").notes` for instructions.

30 datasets out of the box:

| Dataset | Train | Val/Test | Classes | Native size | Source | Notes |
|---------|------:|----------:|--------:|-------------|--------|-------|
| aircraft | 6,667 | 3,333 | 100 | variable | FGVCAircraft | trainval / test split |
| birdsnap | ~25,000 | ~24,829 | 500 | variable | ImageFolder | manual download required |
| caltech101 | ~3,060 | ~5,587 | 101 | variable | Caltech101 | 30 samples/class for train |
| cars | 8,144 | 8,041 | 196 | variable | StanfordCars | may require Kaggle download |
| cifar10 | 50,000 | 10,000 | 10 | 32x32x3 | CIFAR10 | |
| cifar100 | 50,000 | 10,000 | 100 | 32x32x3 | CIFAR100 | |
| clevr | 70,000 | 15,000 | 11 | 320x240x3 | CLEVRClassification | object count (0-10) |
| country211 | 42,200 | 21,100 | 211 | variable | Country211 | train+valid / test |
| cub | 5,994 | 5,794 | 200 | variable | HuggingFace | CUB-200-2011 |
| dtd | 3,760 | 1,880 | 47 | variable | DTD | train+val / test |
| eurosat | 10,000 | 5,000 | 10 | 64x64x3 | EuroSAT | 1000+500 per class |
| fashionmnist | 60,000 | 10,000 | 10 | 28x28x1 | FashionMNIST | |
| fer2013 | 28,709 | 3,589 | 7 | 48x48x1 | FER2013 | Kaggle download required |
| flowers | 2,040 | 6,149 | 102 | variable | Flowers102 | train+val / test |
| food101 | 75,750 | 25,250 | 101 | variable | Food101 | |
| gtsrb | 26,640 | 12,630 | 43 | variable | GTSRB | |
| hatefulmemes | ~8,500 | ~500 | 2 | variable | ImageFolder | Kaggle download required |
| imagenet | 1,281,167 | 50,000 | 1,000 | variable | HuggingFace | gated, requires HF login |
| imagenette | 9,469 | 3,925 | 10 | variable | Imagenette | ImageNet subset |
| kinetics700 | varies | varies | 700 | variable | ImageFolder | frame extraction, ~1.5 days |
| kitti | varies | varies | 4 | variable | ImageFolder | manual prep required |
| mnist | 60,000 | 10,000 | 10 | 28x28x1 | MNIST | |
| pcam | 294,912 | 32,768 | 2 | 96x96x3 | PCAM | train+val / test |
| pets | 3,680 | 3,669 | 37 | variable | OxfordIIITPet | |
| resisc45 | 25,200 | 6,300 | 45 | 256x256x3 | HuggingFace | remote sensing |
| sst | 7,792 | 1,821 | 2 | variable | RenderedSST2 | train+val / test |
| stl10 | 5,000 | 8,000 | 10 | 96x96x3 | STL10 | |
| sun397 | ~19,850 | ~19,850 | 397 | variable | SUN397 | manual download required |
| ucf101 | varies | varies | 101 | variable | ImageFolder | frame extraction required |

Train/test counts reflect the splits as loaded by datavault (some datasets merge train+val for training).
"variable" means images have different native resolutions — all are resized by the transform (default 224x224).
