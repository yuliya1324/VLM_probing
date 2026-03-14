#!/usr/bin/env python3
"""Merge VRD and synthetic representations into one mixed NPZ.

Example:
    python scripts/make_mixed_npz.py \
        --vrd results/vrd_spatial/qwen2/correct/representations.npz \
        --synthetic results/spatial/qwen2/representations.npz \
        --output results/vrd_spatial/qwen2/mixed/representations.npz

    python scripts/make_mixed_npz.py \
        --vrd results/vrd_color/qwen2/correct/representations.npz \
        --synthetic results/color/qwen2/representations.npz \
        --output results/vrd_color/qwen2/mixed/representations.npz
        
bfr merging:
VRD
file: results/vrd_color/qwen2/representations.npz
keys: ['representations', 'labels', 'sample_ids']
n_samples: 2244
unique labels: ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
============================================================
file: results/vrd_shape/qwen2/representations.npz
keys: ['representations', 'labels', 'sample_ids']
n_samples: 1254
unique labels: ['circular', 'oval', 'rectangular', 'square', 'triangular']
============================================================
file: results/vrd_spatial/qwen2/representations.npz
keys: ['representations', 'labels', 'sample_ids']
n_samples: 3284
unique labels: ['above', 'below', 'left of', 'right of']

Syntethic
============================================================
file: results/color/qwen2/representations.npz
keys: ['representations', 'labels', 'image_ids']
n_samples: 2000
unique labels: ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
============================================================
file: results/shape/qwen2/representations.npz
keys: ['representations', 'labels', 'image_ids']
n_samples: 1000
unique labels: ['circular', 'oval', 'rectangular', 'square', 'triangular']
============================================================
file: results/spatial/qwen2/representations.npz
keys: ['representations', 'labels', 'image_ids']
n_samples: 3000
unique labels: ['above', 'below', 'left_of', 'right_of']

-> unify the labels (right_of, left_of) and keys (sample_ids)
"""

import argparse
from pathlib import Path

import numpy as np


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize label variants to a unified form."""
    label_map = {
        "left of": "left_of",
        "right of": "right_of",
        "left_of": "left_of",
        "right_of": "right_of",
        "above": "above",
        "below": "below",
    }
    normalized = [label_map.get(str(x), str(x)) for x in labels.tolist()]
    return np.array(normalized, dtype="<U32")


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)

    if "representations" not in data.files:
        raise ValueError(f"{path}: missing key 'representations'")
    if "labels" not in data.files:
        raise ValueError(f"{path}: missing key 'labels'")

    reps = data["representations"]
    labels = data["labels"]

    if "sample_ids" in data.files:
        ids = data["sample_ids"]
    elif "image_ids" in data.files:
        ids = data["image_ids"]
    else:
        raise ValueError(f"{path}: missing both 'sample_ids' and 'image_ids'")

    labels = normalize_labels(labels)
    ids = np.array([str(x) for x in ids.tolist()], dtype="<U128")

    return reps, labels, ids


def main():
    parser = argparse.ArgumentParser(
        description="Merge VRD and synthetic representations into one mixed NPZ"
    )
    parser.add_argument("--vrd", required=True, help="Path to VRD NPZ")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic NPZ")
    parser.add_argument("--output", required=True, help="Output mixed NPZ path")
    args = parser.parse_args()

    vrd_reps, vrd_labels, vrd_ids = load_npz(args.vrd)
    syn_reps, syn_labels, syn_ids = load_npz(args.synthetic)

    print("=" * 60)
    print("Loaded inputs")
    print("VRD")
    print("  representations:", vrd_reps.shape, vrd_reps.dtype)
    print("  labels         :", vrd_labels.shape, vrd_labels.dtype)
    print("  sample_ids     :", vrd_ids.shape, vrd_ids.dtype)
    print("  unique labels  :", sorted(set(vrd_labels.tolist())))

    print("Synthetic")
    print("  representations:", syn_reps.shape, syn_reps.dtype)
    print("  labels         :", syn_labels.shape, syn_labels.dtype)
    print("  sample_ids     :", syn_ids.shape, syn_ids.dtype)
    print("  unique labels  :", sorted(set(syn_labels.tolist())))

    if vrd_reps.ndim != 3 or syn_reps.ndim != 3:
        raise ValueError("Both representations must be 3D arrays: (N, L, D)")

    if vrd_reps.shape[1:] != syn_reps.shape[1:]:
        raise ValueError(
            f"Representation shape mismatch: "
            f"VRD {vrd_reps.shape[1:]} vs synthetic {syn_reps.shape[1:]}"
        )

    vrd_label_set = set(vrd_labels.tolist())
    syn_label_set = set(syn_labels.tolist())
    if vrd_label_set != syn_label_set:
        raise ValueError(
            "Label space mismatch after normalization:\n"
            f"  only in VRD: {sorted(vrd_label_set - syn_label_set)}\n"
            f"  only in synthetic: {sorted(syn_label_set - vrd_label_set)}"
        )

    mixed_reps = np.concatenate([vrd_reps, syn_reps], axis=0)
    mixed_labels = np.concatenate([vrd_labels, syn_labels], axis=0)
    mixed_ids = np.concatenate([vrd_ids, syn_ids], axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        representations=mixed_reps,
        labels=mixed_labels,
        sample_ids=mixed_ids,
    )

    print("=" * 60)
    print("Saved mixed NPZ")
    print("output         :", output_path)
    print("representations:", mixed_reps.shape, mixed_reps.dtype)
    print("labels         :", mixed_labels.shape, mixed_labels.dtype)
    print("sample_ids     :", mixed_ids.shape, mixed_ids.dtype)
    print("unique labels  :", sorted(set(mixed_labels.tolist())))


if __name__ == "__main__":
    main()