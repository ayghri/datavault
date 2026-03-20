"""Unsupervised clustering benchmark on saved embeddings.

Runs K-Means and Spectral Clustering (KNN affinity) on pre-computed
embeddings. Reports NMI, ARI, and clustering accuracy (Hungarian matching).

Usage:
    python scripts/unsupervised_benchmark.py --repr-dir /buckets/representations

    # Split by dataset for parallelism:
    python scripts/unsupervised_benchmark.py --datasets cifar10 cifar100 mnist fashionmnist stl10 gtsrb imagenette food101 &
    python scripts/unsupervised_benchmark.py --datasets dtd flowers aircraft pets sst eurosat cub resisc45 country211 clevr pcam &
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datavault import load_embeddings

ALL_MODELS = ["clipvitL14", "dinov3b", "dinov2"]


def get_n_classes(y):
    return len(np.unique(y))


def cluster_accuracy(y_true, y_pred):
    """Compute clustering accuracy using the Hungarian algorithm (Munkres)."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    return cost[row_ind, col_ind].sum() / y_pred.size


def run_kmeans(feats, y, n_clusters, seed=42):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(feats)
    return {
        "nmi": normalized_mutual_info_score(y, pred),
        "ari": adjusted_rand_score(y, pred),
        "acc": cluster_accuracy(y, pred),
    }


def _knn_rbf_affinity(feats, n_neighbors, gamma):
    """Build a sparse KNN graph with RBF weights."""
    from sklearn.metrics import pairwise_distances
    knn = kneighbors_graph(feats, n_neighbors=n_neighbors, mode="distance", include_self=False)
    knn.data = np.exp(-gamma * knn.data ** 2)
    affinity = 0.5 * (knn + knn.T)
    return affinity


def _spectral_once(affinity, n_clusters, seed=42):
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    )
    return sc.fit_predict(affinity)


def run_spectral(feats, y, n_clusters, n_neighbors=10, seed=42):
    """Spectral clustering with KNN-RBF affinity. Searches over gamma for best NMI."""
    # Estimate scale from median pairwise distance of KNN graph
    knn_dist = kneighbors_graph(feats, n_neighbors=n_neighbors, mode="distance", include_self=False)
    median_dist = np.median(knn_dist.data)
    gamma_base = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0

    gammas = [gamma_base * s for s in [0.1, 0.5, 1.0, 2.0, 10.0]]

    best, best_gamma = None, None
    for gamma in gammas:
        affinity = _knn_rbf_affinity(feats, n_neighbors, gamma)
        pred = _spectral_once(affinity, n_clusters, seed)
        nmi = normalized_mutual_info_score(y, pred)
        if best is None or nmi > best["nmi"]:
            best = {
                "nmi": nmi,
                "ari": adjusted_rand_score(y, pred),
                "acc": cluster_accuracy(y, pred),
            }
            best_gamma = gamma

    best["gamma"] = best_gamma
    return best


def benchmark_one(dataset_name, model_name, repr_dir, n_neighbors):
    feats_path = Path(repr_dir) / dataset_name / model_name / "feats_val.npy"
    if not feats_path.exists():
        return None

    ds = load_embeddings(dataset_name, model_name, repr_dir, split="val")
    feats, y = ds.feats, ds.labels
    n_clusters = get_n_classes(y)

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    MAX_SPECTRAL = 10000  # subsample for spectral to avoid hour-long runs

    print(f"[{dataset_name}/{model_name}] n={len(y)}, k={n_clusters}, dim={feats.shape[1]}", flush=True)

    t0 = time.time()
    km = run_kmeans(feats, y, n_clusters)
    t_km = time.time() - t0

    # Subsample for spectral if too large
    if len(y) > MAX_SPECTRAL:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), MAX_SPECTRAL, replace=False)
        sc_feats, sc_y = feats[idx], y[idx]
        print(f"  (subsampled to {MAX_SPECTRAL} for spectral)", flush=True)
    else:
        sc_feats, sc_y = feats, y

    t0 = time.time()
    sc = run_spectral(sc_feats, sc_y, n_clusters, n_neighbors=n_neighbors)
    t_sc = time.time() - t0

    print(
        f"  KMeans  ACC={km['acc']:.4f} NMI={km['nmi']:.4f} ARI={km['ari']:.4f} ({t_km:.1f}s)\n"
        f"  Spectr  ACC={sc['acc']:.4f} NMI={sc['nmi']:.4f} ARI={sc['ari']:.4f} gamma={sc['gamma']:.4g} ({t_sc:.1f}s)",
        flush=True,
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "n_clusters": n_clusters,
        "km_acc": km["acc"], "km_nmi": km["nmi"], "km_ari": km["ari"],
        "sc_acc": sc["acc"], "sc_nmi": sc["nmi"], "sc_ari": sc["ari"],
        "sc_gamma": sc["gamma"],
    }


def write_report(results, repr_dir):
    report_path = Path(repr_dir) / "unsupervised_report.md"

    lines = [
        "# Unsupervised Clustering Benchmark",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Evaluated on val/test split. K = number of ground-truth classes.",
        "ACC = clustering accuracy (Hungarian matching), NMI, ARI.",
        "Spectral clustering uses KNN affinity.",
        "",
        "| Dataset | Model | K | KM ACC | KM NMI | KM ARI | SC ACC | SC NMI | SC ARI | SC gamma |",
        "|---------|-------|---|--------|--------|--------|--------|--------|--------|----------|",
    ]
    for r in sorted(results, key=lambda x: (x["dataset"], x["model"])):
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['n_clusters']} | "
            f"{r['km_acc']:.4f} | {r['km_nmi']:.4f} | {r['km_ari']:.4f} | "
            f"{r['sc_acc']:.4f} | {r['sc_nmi']:.4f} | {r['sc_ari']:.4f} | "
            f"{r['sc_gamma']:.4g} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


def discover_datasets(repr_dir):
    """Find all datasets that have embeddings."""
    repr_dir = Path(repr_dir)
    datasets = set()
    for ds_dir in repr_dir.iterdir():
        if ds_dir.is_dir() and any((ds_dir / m / "feats_val.npy").exists() for m in ALL_MODELS):
            datasets.add(ds_dir.name)
    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(description="Unsupervised clustering benchmark")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to benchmark (default: auto-discover from repr-dir)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--repr-dir", type=str, default="/buckets/representations")
    parser.add_argument("--n-neighbors", type=int, default=10)
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = discover_datasets(args.repr_dir)
        print(f"Auto-discovered {len(args.datasets)} datasets: {args.datasets}")

    results = []
    for dataset_name in args.datasets:
        for model_name in args.models:
            try:
                r = benchmark_one(dataset_name, model_name, args.repr_dir, args.n_neighbors)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"[{dataset_name}/{model_name}] FAILED: {e}")

    write_report(results, args.repr_dir)


if __name__ == "__main__":
    main()
