"""Extract ImageNet embeddings using local ImageFolder data.

Usage:
    python scripts/extract_imagenet.py --model clipvitL14 --device cuda:0
    python scripts/extract_imagenet.py --model dinov2 --device cuda:1
    python scripts/extract_imagenet.py --model dinov3b --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datavault import load_model, extract, get_default_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--imagenet-dir", type=str, default="/buckets/datasets/torchvision/imagenet")
    parser.add_argument("--repr-dir", type=str, default="/buckets/representations")
    parser.add_argument("--models-dir", type=str, default="/buckets/models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.repr_dir) / "imagenet" / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "feats_val.npy").exists() and not args.force:
        print(f"Already exists: {save_dir}. Use --force to re-extract.")
        return

    device = torch.device(args.device)
    print(f"Loading model {args.model} on {device}...")
    model, preprocess = load_model(args.model, device=device, models_dir=args.models_dir)
    if preprocess is None:
        preprocess = get_default_transforms()

    imagenet_dir = Path(args.imagenet_dir)
    train_ds = dsets.ImageFolder(str(imagenet_dir / "train"), transform=preprocess)
    val_ds = dsets.ImageFolder(str(imagenet_dir / "val"), transform=preprocess)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    t0 = time.time()
    extract(train_loader, model, device, save_dir=save_dir, suffix="train")
    extract(val_loader, model, device, save_dir=save_dir, suffix="val")
    print(f"[imagenet/{args.model}] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
