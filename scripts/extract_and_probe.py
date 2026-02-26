"""Extract hidden states from a VLM and train probes.

This is the main entry point for step 2 of the pipeline.
Step 1: python scripts/generate_dataset.py --config configs/spatial_dataset.yaml
Step 2: python scripts/extract_and_probe.py  (this script)

Usage:
    # Spatial task with Qwen2-VL
    python scripts/extract_and_probe.py \
        --task spatial \
        --data_dir data/raw/spatial \
        --model_tag qwen2 \
        --output_dir results/qwen2_spatial

    # Color sanity check with LLaVA-1.5
    python scripts/extract_and_probe.py \
        --task color \
        --data_dir data/raw/color \
        --model_tag llava15 \
        --output_dir results/llava15_color

    # Quick test run (10 samples)
    python scripts/extract_and_probe.py \
        --task spatial \
        --data_dir data/raw/spatial \
        --model_tag qwen2 \
        --output_dir results/qwen2_spatial_test \
        --limit 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.extract import extract_dataset
from src.probing.probe import train_probes


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM hidden states and train linear probes"
    )

    # Data
    parser.add_argument("--task", type=str, required=True, choices=["spatial", "color"],
                        help="Task type (determines prompt template and labels)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with metadata.json and images/ from generate_dataset.py")
    parser.add_argument("--train_split_path", type=str, default=None,
                    help="Json file with metadata for train split")
    parser.add_argument("--val_split_path", type=str, default=None,
                    help="Json file with metadata for val spli")

    # Model
    parser.add_argument("--model_tag", type=str, required=True,
                        help="Model tag: qwen2, llava15, etc.")
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID (defaults to registry default for model_tag)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save representations and probe results")

    # Options
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to extract (None = all)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip extraction, use existing .npz (for re-running probes only)")

    # Probe hyperparameters
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse L2 regularization strength")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = str(output_dir / "representations.npz")

    # --- Step 1: Extract ---
    if not args.skip_extraction:
        print("=" * 60)
        print(f"EXTRACTING: task={args.task}  model={args.model_tag}")
        print("=" * 60)

        extract_dataset(
            metadata_path=str(data_dir / "metadata.json"),
            images_dir=str(data_dir / "images"),
            output_path=npz_path,
            model_tag=args.model_tag,
            model_id=args.model_id,
            task=args.task,
            limit=args.limit,
        )
    else:
        if not Path(npz_path).exists():
            print(f"Error: --skip_extraction but {npz_path} not found")
            sys.exit(1)
        print(f"Skipping extraction, using existing: {npz_path}")

    # --- Step 2: Probe ---
    print()
    print("=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)

    results = train_probes(
        representations_path=npz_path,
        output_dir=str(output_dir),
        train_split_path=args.train_split_path,
        val_split_path=args.val_split_path,
        C=args.C,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    best_layer = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest layer: {best_layer} (accuracy: {results[best_layer]['accuracy']:.4f})")


if __name__ == "__main__":
    main()