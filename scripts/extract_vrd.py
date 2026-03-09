"""
Extract hidden states from VRD dataset with periodic checkpointing.

Usage:
    python scripts/extract_vrd.py \
        --csv_path data/vrd_relationships.csv \
        --output_path results/qwen2_vrd/representations.npz \
        --model_tag qwen2 \
        --save_every 200
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.extract import MODEL_REGISTRY, extract_single


VRD_SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{subj}' relative to '{obj}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below]\n"
    "Respond with ONLY the label. No explanation."
)


def build_vrd_prompt(row):
    subj = str(row["subj"])
    obj = str(row["obj"])
    return VRD_SPATIAL_PROMPT.format(subj=subj, obj=obj)


def save_chunk(chunk_reprs, chunk_labels, chunk_ids, chunk_dir, chunk_idx):
    if not chunk_reprs:
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.npz"

    representations = np.stack(chunk_reprs, axis=0)
    labels = np.array(chunk_labels)
    image_ids = np.array(chunk_ids)

    np.savez(
        chunk_path,
        representations=representations,
        labels=labels,
        image_ids=image_ids,
    )
    print(f"Saved chunk: {chunk_path} {representations.shape}")


def merge_chunks(chunk_dir, output_path):
    chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise RuntimeError("No chunk files found to merge.")

    all_reprs = []
    all_labels = []
    all_ids = []

    for chunk_file in chunk_files:
        data = np.load(chunk_file, allow_pickle=True)
        all_reprs.append(data["representations"])
        all_labels.append(data["labels"])
        all_ids.append(data["image_ids"])

    representations = np.concatenate(all_reprs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    image_ids = np.concatenate(all_ids, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        representations=representations,
        labels=labels,
        image_ids=image_ids,
    )

    print(f"\nMerged and saved final file: {output_path}")
    print(f"Final shape: {representations.shape}")
    print(f"{len(labels)} samples saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_tag", type=str, required=True, choices=["qwen2", "llava15", "vila"])
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    output_path = Path(args.output_path)
    chunk_dir = output_path.parent / f"{output_path.stem}_chunks"

    processed_ids = set()
    if args.resume and chunk_dir.exists():
        for chunk_file in sorted(chunk_dir.glob("chunk_*.npz")):
            data = np.load(chunk_file, allow_pickle=True)
            processed_ids.update(map(str, data["image_ids"]))
        print(f"Resume mode: found {len(processed_ids)} already processed samples")

    registry_entry = MODEL_REGISTRY[args.model_tag]
    model_id = args.model_id or registry_entry["default_id"]

    print(f"Loading model: {model_id}")
    model, processor = registry_entry["loader"](model_id)

    chunk_reprs = []
    chunk_labels = []
    chunk_ids = []

    n = len(df)
    saved_chunks = len(list(chunk_dir.glob("chunk_*.npz"))) if chunk_dir.exists() else 0

    for i, (_, row) in enumerate(df.iterrows()):
        image_id = str(row["img_path"])

        if args.resume and image_id in processed_ids:
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"[{i+1}/{n}] already done — skipping")
            continue

        try:
            image_path = Path(row["img_path"])
            image = Image.open(image_path).convert("RGB")
            prompt = build_vrd_prompt(row)
            label = row["relationship"]

            repr_array = extract_single(model, processor, args.model_tag, image, prompt)

            chunk_reprs.append(repr_array)
            chunk_labels.append(label)
            chunk_ids.append(image_id)

            if len(chunk_reprs) >= args.save_every:
                save_chunk(chunk_reprs, chunk_labels, chunk_ids, chunk_dir, saved_chunks)
                saved_chunks += 1
                chunk_reprs, chunk_labels, chunk_ids = [], [], []

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"[{i+1}/{n}] extracted")

        except torch.cuda.OutOfMemoryError:
            print(f"[{i+1}/{n}] OOM — skipping")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{i+1}/{n}] error: {e} — skipping")

    # save remaining samples
    if chunk_reprs:
        save_chunk(chunk_reprs, chunk_labels, chunk_ids, chunk_dir, saved_chunks)

    merge_chunks(chunk_dir, output_path)


if __name__ == "__main__":
    main()