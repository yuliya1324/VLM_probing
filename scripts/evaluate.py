#!/usr/bin/env python3
"""Evaluate trained probes across ALL layers on a dataset and plot accuracy.

Supports two data formats:
  1. Synthetic (.npz) — from our extract_and_probe.py pipeline
  2. Collaborator's .pt files — from extract_qwen2.py / extract_spatialRGBT.py

Usage:
    # Evaluate using .npz (synthetic dataset)
    python scripts/evaluate_all_layers.py \
        --probes_dir results/qwen2_spatial/probes \
        --representations data/processed/qwen2_spatial.npz \
        --output results/qwen2_spatial/eval_all_layers.png

    # Evaluate using .pt files (collaborator's VRD format)
    python scripts/evaluate_all_layers.py \
        --probes_dir results/qwen2_vrd/probes \
        --pt_dir features/Qwen2-VL \
        --output results/qwen2_vrd/eval_all_layers.png

    # Compare multiple runs
    python scripts/evaluate_all_layers.py \
        --probes_dir results/qwen2_spatial/probes results/vila_spatial/probes \
        --representations results/qwen2_spatial/representations.npz results/vila_spatial/representations.npz \
        --labels "Qwen2-VL" "SpatialRGPT-VILA" \
        --output results/comparison.png
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing.probe import load_probe


# ============================================================
# Data loading
# ============================================================

def load_from_npz(npz_path: str, split: str = "val", train_ratio: float = 0.8, seed: int = 42):
    """Load representations and labels from .npz, return val split only."""
    data = np.load(npz_path, allow_pickle=True)
    representations = data["representations"]  # (n_samples, n_layers, hidden_dim)
    labels = data["labels"]
    image_ids = data["image_ids"]

    n = len(labels)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    split_point = int(n * train_ratio)

    if split == "val":
        idx = indices[split_point:]
    else:
        idx = indices[:split_point]

    return representations[idx], labels[idx], image_ids[idx]


def load_from_pt_dir(pt_dir: str):
    """Load representations and labels from a directory of .pt files.

    Each .pt file has:
        {"layers": {0: tensor, 1: tensor, ...}, "meta": {"rel": ..., ...}}
    """
    pt_dir = Path(pt_dir)
    pt_files = sorted(pt_dir.glob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")

    import torch

    all_reprs = []
    all_labels = []
    all_ids = []

    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        layers = data["layers"]
        meta = data["meta"]

        label = meta.get("rel") or meta.get("relationship")
        if label is None:
            continue

        n_layers = len(layers)
        layer_tensors = [layers[l].float().numpy() for l in range(n_layers)]
        sample_repr = np.stack(layer_tensors, axis=0)

        all_reprs.append(sample_repr)
        all_labels.append(label)
        all_ids.append(pt_path.stem)

    representations = np.stack(all_reprs, axis=0)
    labels = np.array(all_labels)
    image_ids = np.array(all_ids)

    print(f"Loaded {len(labels)} samples from .pt files ({n_layers} layers, {representations.shape[2]} dim)")
    return representations, labels, image_ids


# ============================================================
# Evaluation
# ============================================================

def evaluate_all_layers(
    probes_dir: str,
    representations: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Run every layer's probe on the data and return per-layer accuracy.

    Args:
        probes_dir: Path to probes/ directory with probe_layer_*.joblib
        representations: (n_samples, n_layers, hidden_dim)
        labels: (n_samples,) string labels

    Returns:
        {"layer_accuracies": [float, ...], "n_layers": int, "n_samples": int, ...}
    """
    probes_dir = Path(probes_dir)

    # Load manifest to get n_layers and class info
    manifest_path = probes_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    n_layers = manifest["n_layers"]
    classes = manifest["classes"]

    # Encode labels using the same encoder
    import joblib
    le = joblib.load(probes_dir / "label_encoder.joblib")
    y_true = le.transform(labels)

    layer_accuracies = []
    per_class_accuracies = []

    for layer_idx in range(n_layers):
        probe_path = probes_dir / f"probe_layer_{layer_idx:03d}.joblib"
        if not probe_path.exists():
            layer_accuracies.append(None)
            per_class_accuracies.append(None)
            continue

        probe = joblib.load(probe_path)
        X = representations[:, layer_idx, :]
        y_pred = probe.predict(X)

        acc = (y_pred == y_true).mean()
        layer_accuracies.append(float(acc))

        # Per-class accuracy
        class_accs = {}
        for cls_idx, cls_name in enumerate(classes):
            mask = y_true == cls_idx
            if mask.sum() > 0:
                class_accs[cls_name] = float((y_pred[mask] == y_true[mask]).mean())
        per_class_accuracies.append(class_accs)

        print(f"  Layer {layer_idx:2d}: {acc:.4f}")

    results = {
        "n_layers": n_layers,
        "n_samples": len(labels),
        "classes": classes,
        "layer_accuracies": layer_accuracies,
        "per_class_accuracies": per_class_accuracies,
        "label_distribution": dict(Counter(labels)),
    }

    best_layer = max(range(n_layers), key=lambda i: layer_accuracies[i] or 0)
    results["best_layer"] = best_layer
    results["best_accuracy"] = layer_accuracies[best_layer]

    print(f"\n  Best: layer {best_layer} ({layer_accuracies[best_layer]:.4f})")
    return results


# ============================================================
# Plotting
# ============================================================

def plot_results(
    all_results: list,
    all_labels: list,
    output_path: str = None,
    title: str = "Probe Accuracy Across Layers",
    show_per_class: bool = False,
):
    """Plot accuracy curves for one or more evaluation runs."""

    if show_per_class and len(all_results) == 1:
        # Single run with per-class breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = None

    # Main accuracy plot
    for results, label in zip(all_results, all_labels):
        accs = results["layer_accuracies"]
        layers = list(range(len(accs)))
        ax1.plot(layers, accs, marker="o", markersize=3, label=label)

        best_l = results["best_layer"]
        best_a = results["best_accuracy"]
        ax1.annotate(
            f"L{best_l} ({best_a:.3f})",
            xy=(best_l, best_a),
            xytext=(best_l + 1, best_a - 0.03),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    n_classes = len(all_results[0]["classes"])
    chance = 1.0 / n_classes
    ax1.axhline(y=chance, color="red", linestyle="--", alpha=0.5,
                label=f"chance ({chance:.2f}, {n_classes} classes)")

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Per-class breakdown (only for single run)
    if ax2 is not None:
        results = all_results[0]
        classes = results["classes"]
        for cls in classes:
            cls_accs = [
                pa[cls] if pa and cls in pa else None
                for pa in results["per_class_accuracies"]
            ]
            layers = list(range(len(cls_accs)))
            ax2.plot(layers, cls_accs, marker="o", markersize=2, label=cls)

        ax2.axhline(y=chance, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Per-Class Accuracy")
        ax2.set_title("Per-Class Breakdown")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot → {output_path}")
    else:
        plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate probes across all layers")

    parser.add_argument("--probes_dir", type=str, nargs="+", required=True,
                        help="Path(s) to probes/ directory")

    # Data source — pick one per run
    parser.add_argument("--representations", type=str, nargs="*", default=None,
                        help=".npz file(s) from extract_and_probe.py")
    parser.add_argument("--pt_dir", type=str, nargs="*", default=None,
                        help="Directory(s) of .pt files for VRD representations")

    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Legend labels for each run")
    parser.add_argument("--output", type=str, default=None,
                        help="Save plot to file")
    parser.add_argument("--title", type=str, default="Probe Accuracy Across Layers")
    parser.add_argument("--per_class", action="store_true",
                        help="Show per-class accuracy breakdown (single run only)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which split to evaluate on for .npz data")
    parser.add_argument("--save_json", action="store_true",
                        help="Save detailed results as JSON alongside the plot")

    args = parser.parse_args()

    n_runs = len(args.probes_dir)

    # Determine data sources
    if args.representations:
        data_sources = [("npz", p) for p in args.representations]
    elif args.pt_dir:
        data_sources = [("pt", p) for p in args.pt_dir]
    else:
        # Try to find representations.npz inside each probes_dir's parent
        data_sources = []
        for pd in args.probes_dir:
            parent = Path(pd).parent
            npz = parent / "representations.npz"
            if npz.exists():
                data_sources.append(("npz", str(npz)))
            else:
                print(f"Error: no data source for {pd}. Use --representations or --pt_dir")
                sys.exit(1)

    if len(data_sources) != n_runs:
        print(f"Error: {n_runs} probe dirs but {len(data_sources)} data sources")
        sys.exit(1)

    # Labels
    if args.labels:
        run_labels = args.labels
    else:
        run_labels = [Path(pd).parent.name for pd in args.probes_dir]

    # Evaluate
    all_results = []
    for i, (probes_dir, (src_type, src_path)) in enumerate(zip(args.probes_dir, data_sources)):
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_labels[i]}")
        print(f"  Probes: {probes_dir}")
        print(f"  Data:   {src_path} ({src_type})")
        print(f"{'='*60}")

        if src_type == "npz":
            representations, labels, image_ids = load_from_npz(src_path, split=args.split)
        else:
            representations, labels, image_ids = load_from_pt_dir(src_path)

        results = evaluate_all_layers(probes_dir, representations, labels)
        all_results.append(results)

        if args.save_json:
            json_path = Path(args.output).with_suffix(".json") if args.output else Path(probes_dir).parent / "eval_results.json"
            # Convert for JSON serialization
            save_results = {k: v for k, v in results.items()}
            with open(json_path, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"  Saved JSON → {json_path}")

    # Plot
    plot_results(all_results, run_labels, args.output, args.title, args.per_class)


if __name__ == "__main__":
    main()