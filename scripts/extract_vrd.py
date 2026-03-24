# scripts/extract_vrd.py

"""
Extract hidden states from VRD CSVs with periodic checkpointing.

Usages:
    python scripts/extract_vrd.py --task spatial --model_tag qwen2
    python scripts/extract_vrd.py --task color --model_tag llava15
    python scripts/extract_vrd.py --task shape --model_tag vila --save_every 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing.vrd import (
    TASK_TO_CSV,
    build_prompt,
    get_image_path,
    get_label,
    get_sample_id,
    load_vrd_dataframe,
)
from src.extraction.extract import MODEL_REGISTRY, extract_single
from src.extraction.io import merge_chunks, save_chunk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

TASK_TO_RESULT_DIR = {
    "spatial": RESULTS_DIR / "vrd" / "spatial",
    "color": RESULTS_DIR / "vrd" / "color",
    "shape": RESULTS_DIR / "vrd" / "shape",
}

# avoid OOM
def resize_for_vrd(image: Image.Image, max_size: int = 448) -> Image.Image:
    """Resize image for VRD extraction while preserving aspect ratio.

    The returned image fits within (max_size, max_size).
    """
    image = image.convert("RGB")
    return ImageOps.contain(image, (max_size, max_size))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["spatial", "color", "shape"])
    parser.add_argument("--model_tag", type=str, required=True, choices=["qwen2", "llava15", "vila"])
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else TASK_TO_CSV[args.task]
    output_path = (
        Path(args.output_path)
        if args.output_path
        else TASK_TO_RESULT_DIR[args.task] / args.model_tag / "representations.npz"
    )
    chunk_dir = output_path.parent / f"{output_path.stem}_chunks"

    print(f"Task       : {args.task}")
    print(f"Model tag  : {args.model_tag}")
    print(f"CSV path   : {csv_path}")
    print(f"Output path: {output_path}")

    df = load_vrd_dataframe(args.task, str(csv_path))

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    processed_ids = set()
    if args.resume and chunk_dir.exists():
        for chunk_file in sorted(chunk_dir.glob("chunk_*.npz")):
            data = np.load(chunk_file, allow_pickle=True)
            key_name = "sample_ids" if "sample_ids" in data.files else "image_ids"
            processed_ids.update(map(str, data[key_name]))
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
        sample_id = get_sample_id(row, args.task)

        if args.resume and sample_id in processed_ids:
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"[{i+1}/{n}] already done — skipping")
            continue

        try:
            image_path = get_image_path(row)
            image = Image.open(image_path)
            image = resize_for_vrd(image, max_size=448)

            prompt = build_prompt(row, args.task)
            label = get_label(row, args.task)

            repr_array = extract_single(model, processor, args.model_tag, image, prompt)

            chunk_reprs.append(repr_array)
            chunk_labels.append(label)
            chunk_ids.append(sample_id)

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

    if chunk_reprs:
        save_chunk(chunk_reprs, chunk_labels, chunk_ids, chunk_dir, saved_chunks)

    merge_chunks(chunk_dir, output_path)


if __name__ == "__main__":
    main()