"""Extract embeddings for multiple datasets/models and benchmark with linear probe.

Saves embeddings to repr_dir/{dataset}/{model}/feats_{split}.npy
and writes a report.md summary in the same folder.

Usage:
    python scripts/extract_and_benchmark.py \
        --datasets cifar10 cifar100 \
        --models clipvitL14 dinov2 dinov3b \
        --repr-dir /buckets/representations \
        --device cuda:0

    # Use both GPUs (run two instances):
    python scripts/extract_and_benchmark.py --datasets cifar10 --models clipvitL14 dinov3b --device cuda:0 &
    python scripts/extract_and_benchmark.py --datasets cifar10 --models dinov2 --device cuda:1 &
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datavault import (
    get_dataloaders,
    get_default_transforms,
    load_model,
    extract,
    load_embeddings,
    list_models,
    list_datasets,
)


def extract_and_benchmark(dataset_name, model_name, args):
    device = torch.device(args.device)
    repr_dir = Path(args.repr_dir) / dataset_name / model_name
    repr_dir.mkdir(parents=True, exist_ok=True)

    feats_train_path = repr_dir / "feats_train.npy"
    feats_val_path = repr_dir / "feats_val.npy"

    # Extract or load cached
    if feats_train_path.exists() and feats_val_path.exists() and not args.force:
        print(f"[{dataset_name}/{model_name}] Loading cached embeddings from {repr_dir}")
        ds_train = load_embeddings(dataset_name, model_name, args.repr_dir, split="train")
        ds_val = load_embeddings(dataset_name, model_name, args.repr_dir, split="val")
        feats_train, y_train = ds_train.feats, ds_train.labels
        feats_val, y_val = ds_val.feats, ds_val.labels
        t_extract = 0.0
    else:
        print(f"[{dataset_name}/{model_name}] Extracting on {args.device}...")
        model, preprocess = load_model(model_name, device=device, models_dir=args.models_dir)
        if preprocess is None:
            preprocess = get_default_transforms()

        train_loader, val_loader = get_dataloaders(
            dataset_name,
            transform=preprocess,
            batch_size=args.batch_size,
            root_dir=args.data_dir,
        )

        t0 = time.time()
        feats_train, y_train = extract(train_loader, model, device, save_dir=repr_dir, suffix="train")
        feats_val, y_val = extract(val_loader, model, device, save_dir=repr_dir, suffix="val")
        t_extract = time.time() - t0

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Benchmark
    scaler = StandardScaler()
    feats_train_scaled = scaler.fit_transform(feats_train)
    feats_val_scaled = scaler.transform(feats_val)

    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs", n_jobs=-1)
    clf.fit(feats_train_scaled, y_train)
    t_clf = time.time() - t0

    train_acc = clf.score(feats_train_scaled, y_train)
    val_acc = clf.score(feats_val_scaled, y_val)

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "dim": feats_train.shape[1],
        "n_train": feats_train.shape[0],
        "n_val": feats_val.shape[0],
        "extract_time": t_extract,
        "clf_time": t_clf,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }

    print(
        f"[{dataset_name}/{model_name}] "
        f"dim={result['dim']} "
        f"train_acc={train_acc:.4f} "
        f"val_acc={val_acc:.4f} "
        f"extract={t_extract:.1f}s"
    )

    # Write per-model report in the model's own folder
    model_report = repr_dir / "report.md"
    model_report.write_text(
        f"# {dataset_name} / {model_name}\n\n"
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- Device: {args.device}\n"
        f"- Embedding dim: {result['dim']}\n"
        f"- Train samples: {result['n_train']}\n"
        f"- Val samples: {result['n_val']}\n"
        f"- Extraction time: {t_extract:.1f}s\n"
        f"- Train accuracy: {train_acc:.4f}\n"
        f"- **Val accuracy: {val_acc:.4f}**\n"
        f"- Classifier (LogReg C={args.C}): {t_clf:.1f}s\n"
    )

    return result


def collect_all_results(repr_dir):
    """Scan repr_dir for all existing embeddings and benchmark them."""
    repr_dir = Path(repr_dir)
    results = []
    for ds_dir in sorted(repr_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            feats_path = model_dir / "feats_val.npy"
            labels_path = model_dir / "y_val.npy"
            if not feats_path.exists() or not labels_path.exists():
                continue
            feats = np.load(feats_path)
            labels = np.load(labels_path)

            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(feats_scaled, labels)
            val_acc = clf.score(feats_scaled, labels)

            # Load train for proper eval
            feats_train_path = model_dir / "feats_train.npy"
            labels_train_path = model_dir / "y_train.npy"
            if feats_train_path.exists() and labels_train_path.exists():
                ft = np.load(feats_train_path)
                yt = np.load(labels_train_path)
                scaler = StandardScaler()
                ft_s = scaler.fit_transform(ft)
                fv_s = scaler.transform(feats)
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
                clf.fit(ft_s, yt)
                train_acc = clf.score(ft_s, yt)
                val_acc = clf.score(fv_s, labels)
            else:
                train_acc = val_acc

            results.append({
                "dataset": ds_dir.name,
                "model": model_dir.name,
                "dim": feats.shape[1],
                "train_acc": train_acc,
                "val_acc": val_acc,
            })
    return results


def write_summary(results, repr_dir):
    """Write/overwrite the summary report.md from the full set of results."""
    if not results:
        return

    summary_path = Path(repr_dir) / "report.md"

    lines = [
        "# Embedding Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| Dataset | Model | Dim | Train Acc | Val Acc |",
        "|---------|-------|-----|-----------|---------|",
    ]
    for r in sorted(results, key=lambda x: (x["dataset"], -x["val_acc"])):
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['dim']} | "
            f"{r['train_acc']:.4f} | **{r['val_acc']:.4f}** |"
        )
    lines.append("")

    summary_path.write_text("\n".join(lines))
    print(f"\nSummary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and benchmark")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"],
                        help=f"Available: {list_datasets()}")
    parser.add_argument("--models", nargs="+", default=["clipvitL14", "dinov2", "dinov3b"],
                        help=f"Available: {list_models()}")
    parser.add_argument("--repr-dir", type=str, default="/buckets/representations")
    parser.add_argument("--data-dir", type=str, default="/buckets/datasets")
    parser.add_argument("--models-dir", type=str, default="/buckets/models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        for model_name in args.models:
            try:
                extract_and_benchmark(dataset_name, model_name, args)
            except Exception as e:
                print(f"[{dataset_name}/{model_name}] FAILED: {e}")

    # Rebuild full report from all cached embeddings
    print("\nRebuilding full report from all cached embeddings...")
    all_results = collect_all_results(args.repr_dir)
    write_summary(all_results, args.repr_dir)


if __name__ == "__main__":
    main()
