"""Run logistic regression, k-means, and spectral clustering benchmarks on pre-computed embeddings.

Usage:
    python scripts/run_report.py --repr-dir /buckets/representations
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_MODELS = ["clipvitL14", "dinov3b", "dinov2"]


def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    return cost[row_ind, col_ind].sum() / y_pred.size


def cluster_metrics(y, pred):
    return {
        "acc": cluster_accuracy(y, pred),
        "nmi": normalized_mutual_info_score(y, pred),
        "ari": adjusted_rand_score(y, pred),
    }


def discover_datasets(repr_dir):
    repr_dir = Path(repr_dir)
    datasets = set()
    for ds_dir in repr_dir.iterdir():
        if ds_dir.is_dir() and any(
            (ds_dir / m / "feats_val.npy").exists() for m in ALL_MODELS
        ):
            datasets.add(ds_dir.name)
    return sorted(datasets)


def run_logreg(feats_train, y_train, feats_val, y_val):
    scaler = StandardScaler()
    ft = scaler.fit_transform(feats_train)
    fv = scaler.transform(feats_val)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(ft, y_train)
    return clf.score(fv, y_val)


def run_kmeans(feats, y, n_clusters, seed=42):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(feats)
    return cluster_metrics(y, pred)


def _knn_rbf_affinity(feats, n_neighbors, gamma):
    knn = kneighbors_graph(feats, n_neighbors=n_neighbors, mode="distance", include_self=False)
    knn.data = np.exp(-gamma * knn.data ** 2)
    affinity = 0.5 * (knn + knn.T)
    return affinity


def run_spectral(feats, y, n_clusters, n_neighbors, seed=42):
    knn_dist = kneighbors_graph(feats, n_neighbors=n_neighbors, mode="distance", include_self=False)
    median_dist = np.median(knn_dist.data)
    gamma_base = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0

    gammas = [gamma_base * s for s in [0.1, 0.5, 1.0, 2.0, 10.0]]

    best, best_gamma = None, None
    for gamma in gammas:
        affinity = _knn_rbf_affinity(feats, n_neighbors, gamma)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=seed,
            assign_labels="kmeans",
        )
        pred = sc.fit_predict(affinity)
        m = cluster_metrics(y, pred)
        if best is None or m["nmi"] > best["nmi"]:
            best = m
            best_gamma = gamma

    best["gamma"] = best_gamma
    return best


def load_embeddings(repr_dir, dataset, model, split):
    base = Path(repr_dir) / dataset / model
    feats = np.load(base / f"feats_{split}.npy")
    labels = np.load(base / f"y_{split}.npy")
    return feats, labels


def compute_knn_k(y):
    """k = min(avg_class_size / 10, 100), at least 2."""
    classes, counts = np.unique(y, return_counts=True)
    avg_size = counts.mean()
    return max(2, min(int(avg_size / 10), 100))


MAX_SPECTRAL = 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-dir", type=str, default="/buckets/representations")
    args = parser.parse_args()

    datasets = discover_datasets(args.repr_dir)
    print(f"Found {len(datasets)} datasets: {datasets}\n")

    logreg_results = {}
    kmeans_results = {}
    spectral_results = {}
    dataset_meta = {}  # dataset -> {n_clusters, knn_k}

    for dataset in datasets:
        logreg_results[dataset] = {}
        kmeans_results[dataset] = {}
        spectral_results[dataset] = {}
        for model in ALL_MODELS:
            base = Path(args.repr_dir) / dataset / model
            if not (base / "feats_val.npy").exists():
                continue
            try:
                ft, yt = load_embeddings(args.repr_dir, dataset, model, "train")
                fv, yv = load_embeddings(args.repr_dir, dataset, model, "val")

                n_clusters = len(np.unique(yv))
                knn_k = compute_knn_k(yv)
                dataset_meta[dataset] = {"n_clusters": n_clusters, "knn_k": knn_k}

                # LogReg
                val_acc = run_logreg(ft, yt, fv, yv)
                logreg_results[dataset][model] = val_acc

                # StandardScaler for clustering (shared)
                scaler = StandardScaler()
                fv_scaled = scaler.fit_transform(fv)

                # K-Means
                km = run_kmeans(fv_scaled, yv, n_clusters)
                kmeans_results[dataset][model] = km

                # Spectral clustering (subsample if too large)
                if len(yv) > MAX_SPECTRAL:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(yv), MAX_SPECTRAL, replace=False)
                    sc_feats, sc_y = fv_scaled[idx], yv[idx]
                    print(f"  (subsampled to {MAX_SPECTRAL} for spectral)", flush=True)
                else:
                    sc_feats, sc_y = fv_scaled, yv

                t0 = time.time()
                sc = run_spectral(sc_feats, sc_y, n_clusters, n_neighbors=knn_k)
                t_sc = time.time() - t0
                spectral_results[dataset][model] = sc

                print(
                    f"[{dataset}/{model}] LogReg={val_acc:.4f}  "
                    f"KM_ACC={km['acc']:.4f}  "
                    f"SC_ACC={sc['acc']:.4f} gamma={sc['gamma']:.4g} k={knn_k} ({t_sc:.1f}s)"
                )
            except Exception as e:
                import traceback
                print(f"[{dataset}/{model}] FAILED: {e}")
                traceback.print_exc()

    # Write report
    report_path = Path(args.repr_dir) / "report.md"
    lines = [
        "# Embedding Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Linear Probe",
        "",
        "LogisticRegression (C=1.0) on val/test split.",
        "",
        "| Dataset | CLIP ViT-L/14 | DINOv3 ViT-B/16 | DINOv2 ViT-g/14 |",
        "|---------|:---:|:---:|:---:|",
    ]
    for dataset in datasets:
        vals = []
        best_val = max(
            (logreg_results[dataset].get(m, -1) for m in ALL_MODELS), default=-1
        )
        for model in ALL_MODELS:
            acc = logreg_results[dataset].get(model)
            if acc is None:
                vals.append("-")
            elif acc == best_val:
                vals.append(f"**{acc:.4f}**")
            else:
                vals.append(f"{acc:.4f}")
        lines.append(f"| {dataset} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # K-Means section
    lines.extend([
        "",
        "## K-Means Clustering",
        "",
        "Evaluated on val/test split. K = number of ground-truth classes.",
        "ACC = clustering accuracy (Hungarian matching), NMI, ARI.",
        "",
        "| Dataset | Model | K | ACC | NMI | ARI |",
        "|---------|-------|---|-----|-----|-----|",
    ])
    for dataset in datasets:
        meta = dataset_meta.get(dataset, {})
        n_clusters = meta.get("n_clusters", "?")
        for model in ALL_MODELS:
            km = kmeans_results[dataset].get(model)
            if km is None:
                continue
            lines.append(
                f"| {dataset} | {model} | {n_clusters} | "
                f"{km['acc']:.4f} | {km['nmi']:.4f} | {km['ari']:.4f} |"
            )

    # Spectral Clustering section
    lines.extend([
        "",
        "## Spectral Clustering (KNN-RBF)",
        "",
        "KNN affinity with RBF kernel, gamma selected over 5 scales.",
        "k = min(avg\\_class\\_size / 10, 100). Subsampled to 10k for large datasets.",
        "",
        "| Dataset | Model | K | k_nn | ACC | NMI | ARI | gamma |",
        "|---------|-------|---|------|-----|-----|-----|-------|",
    ])
    for dataset in datasets:
        meta = dataset_meta.get(dataset, {})
        n_clusters = meta.get("n_clusters", "?")
        knn_k = meta.get("knn_k", "?")
        for model in ALL_MODELS:
            sc = spectral_results[dataset].get(model)
            if sc is None:
                continue
            lines.append(
                f"| {dataset} | {model} | {n_clusters} | {knn_k} | "
                f"{sc['acc']:.4f} | {sc['nmi']:.4f} | {sc['ari']:.4f} | "
                f"{sc['gamma']:.4g} |"
            )
    lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
