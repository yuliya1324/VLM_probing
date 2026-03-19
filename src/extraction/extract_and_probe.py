"""Extract hidden states from a VLM and train probes.

Usage:
    # Standard pipeline: extract + train
    python scripts/extract_and_probe.py \
        --task spatial \
        --data_dir data/raw/spatial \
        --model_tag qwen2 \
        --output_dir results/qwen2_spatial

    # Re-train probes from an existing NPZ
    python scripts/extract_and_probe.py \
        --task color \
        --model_tag qwen2 \
        --output_dir results/vrd_color/qwen2/correct \
        --skip_extraction

    # Re-train probes from an explicitly specified NPZ
    python scripts/extract_and_probe.py \
        --task color \
        --model_tag qwen2 \
        --output_dir results/vrd_color/qwen2/correct_retrain \
        --skip_extraction \
        --representations_path results/vrd_color/qwen2/correct/representations.npz
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

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["spatial", "color", "shape"],
        help="Task type (determines prompt template and labels)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with metadata.json and images/. Required unless --skip_extraction is used.",
    )
    parser.add_argument(
        "--train_split_path",
        type=str,
        default=None,
        help="JSON file with metadata for train split",
    )
    parser.add_argument(
        "--val_split_path",
        type=str,
        default=None,
        help="JSON file with metadata for val split",
    )

    parser.add_argument(
        "--model_tag",
        type=str,
        required=True,
        help="Model tag: qwen2, llava15, etc.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace model ID (defaults to registry default for model_tag)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save representations and probe results",
    )
    parser.add_argument(
        "--representations_path",
        type=str,
        default=None,
        help="Optional path to an existing representations.npz. "
             "If omitted, uses <output_dir>/representations.npz",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to extract (None = all)",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip extraction and use an existing .npz",
    )
    parser.add_argument(
        "--random_prompt",
        action="store_true",
        help="Use random prompt",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.representations_path) if args.representations_path else output_dir / "representations.npz"

    if not args.skip_extraction:
        if args.data_dir is None:
            parser.error("--data_dir is required unless --skip_extraction is used.")

        data_dir = Path(args.data_dir)

        print("=" * 60)
        print(f"EXTRACTING: task={args.task}  model={args.model_tag}")
        print("=" * 60)

        extract_dataset(
            metadata_path=str(data_dir / "metadata.json"),
            images_dir=str(data_dir / "images"),
            output_path=str(npz_path),
            model_tag=args.model_tag,
            model_id=args.model_id,
            task=args.task,
            limit=args.limit,
            random_prompt=args.random_prompt,
        )
    else:
        if not npz_path.exists():
            parser.error(f"--skip_extraction was set, but NPZ not found: {npz_path}")
        print(f"Skipping extraction, using existing: {npz_path}")

    print()
    print("=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)

    results = train_probes(
        representations_path=str(npz_path),
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