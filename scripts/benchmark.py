"""Benchmark embedding models with a linear classifier.

Usage:
    python scripts/benchmark.py --dataset cifar10 --models clipvitL14 dinov2
    python scripts/benchmark.py --dataset cifar100 --models dinov2
"""

import argparse
import time
import sys
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Support running as: python scripts/benchmark.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datavault import (
    get_dataloaders,
    get_default_transforms,
    load_model,
    extract,
    list_models,
    list_datasets,
)


def run(args):
    device = torch.device(args.device)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset} | Model: {model_name}")
        print(f"{'='*60}")

        model, preprocess = load_model(model_name, device=device, models_dir=args.models_dir)
        if preprocess is None:
            preprocess = get_default_transforms()

        train_loader, val_loader = get_dataloaders(
            args.dataset,
            transform=preprocess,
            batch_size=args.batch_size,
            root_dir=args.data_dir,
        )

        t0 = time.time()
        feats_train, y_train = extract(train_loader, model, device)
        feats_val, y_val = extract(val_loader, model, device)
        t_extract = time.time() - t0

        print(f"Features: train {feats_train.shape}, val {feats_val.shape}")
        print(f"Extraction time: {t_extract:.1f}s")

        scaler = StandardScaler()
        feats_train = scaler.fit_transform(feats_train)
        feats_val = scaler.transform(feats_val)

        t0 = time.time()
        clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs", n_jobs=-1)
        clf.fit(feats_train, y_train)
        t_train = time.time() - t0

        train_acc = clf.score(feats_train, y_train)
        val_acc = clf.score(feats_val, y_val)

        print(f"Train acc: {train_acc:.4f}")
        print(f"Val acc:   {val_acc:.4f}")
        print(f"Classifier training time: {t_train:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models with linear probe")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help=f"Dataset name. Available: {list_datasets()}")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression regularization")
    parser.add_argument("--models", nargs="+", default=["dinov2", "clipvitL14"],
                        help=f"Models to benchmark. Available: {list_models()}")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
