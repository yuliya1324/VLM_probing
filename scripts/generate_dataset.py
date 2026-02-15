#!/usr/bin/env python3
"""Generate synthetic datasets for VLM spatial probing.

Usage:
    python scripts/generate_dataset.py --config configs/spatial_dataset.yaml
    python scripts/generate_dataset.py --config configs/color_dataset.yaml
    python scripts/generate_dataset.py --task spatial --n_samples 500  # quick test
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_generation.spatial import generate_spatial_dataset
from src.dataset_generation.color import generate_color_dataset


def split_dataset(metadata_path: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """Create train/val splits from a metadata.json file."""
    with open(metadata_path) as f:
        samples = json.load(f)

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    split_point = int(len(indices) * train_ratio)
    train_ids = indices[:split_point]
    val_ids = indices[split_point:]

    splits_dir = Path(output_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [samples[i] for i in train_ids]
    val_samples = [samples[i] for i in val_ids]

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"Split: {len(train_samples)} train, {len(val_samples)} val → {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic probing datasets")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--task", type=str, choices=["spatial", "color"], help="Task (overrides config)")
    parser.add_argument("--n_samples", type=int, help="Number of samples (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--no_split", action="store_true", help="Skip train/val splitting")
    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # CLI args override config
    task = args.task or cfg.get("task", "spatial")
    n_samples = args.n_samples or cfg.get("n_samples", 3000)
    seed = args.seed or cfg.get("seed", 42)
    output_dir = args.output_dir or cfg.get("output_dir", f"data/raw/{task}")

    print(f"Task: {task} | Samples: {n_samples} | Seed: {seed} | Output: {output_dir}")

    if task == "spatial":
        prompt_idx = cfg.get("prompt_template_index", None)
        samples = generate_spatial_dataset(
            n_samples=n_samples,
            output_dir=output_dir,
            seed=seed,
            prompt_template_index=prompt_idx,
        )
    elif task == "color":
        prompt_idx = cfg.get("prompt_template_index", None)
        samples = generate_color_dataset(
            n_samples=n_samples,
            output_dir=output_dir,
            seed=seed,
            min_shapes=cfg.get("min_shapes", 1),
            max_shapes=cfg.get("max_shapes", 3),
            prompt_template_index=prompt_idx,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # Print class distribution
    if task == "spatial":
        label_key = "relation"
    else:
        label_key = "color_label"

    from collections import Counter
    dist = Counter(s[label_key] for s in samples)
    print(f"Class distribution: {dict(dist)}")

    # Create train/val splits
    if not args.no_split:
        meta_path = Path(output_dir) / "metadata.json"
        splits_dir = f"data/splits/{task}"
        split_dataset(str(meta_path), splits_dir, seed=seed)


if __name__ == "__main__":
    main()