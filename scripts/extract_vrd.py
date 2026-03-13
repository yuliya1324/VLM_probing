"""
Extract hidden states from VRD CSVs with periodic checkpointing.

Usages:
    python scripts/extract_vrd.py --task spatial --model_tag qwen2
    python scripts/extract_vrd.py --task color --model_tag llava15
    python scripts/extract_vrd.py --task shape --model_tag vila --save_every 200
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


# ============================================================
# Prompts
# ============================================================
SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{subj}' relative to '{obj}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below]\n"
    "Respond with ONLY the label. No explanation."
)

COLOR_PROMPT = (
    "What is the color of the {subj} in the image?\n"
    "Respond with ONLY the color name. No explanation."
)

SHAPE_PROMPT = (
    "What is the shape of the {subj} in the image?\n"
    "Choose ONE label from:\n"
    "[circular, oval, square, rectangular, triangular]\n"
    "Respond with ONLY the label. No explanation."
)

PROMPT_TEMPLATES = {
    "spatial": SPATIAL_PROMPT,
    "color": COLOR_PROMPT,
    "shape": SHAPE_PROMPT,
}


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vrd_csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TASK_TO_CSV = {
    "spatial": DATA_DIR / "vrd_spatial.csv",
    "color": DATA_DIR / "vrd_color.csv",
    "shape": DATA_DIR / "vrd_shape.csv",
}

TASK_TO_RESULT_DIR = {
    "spatial": RESULTS_DIR / "vrd_spatial",
    "color": RESULTS_DIR / "vrd_color",
    "shape": RESULTS_DIR / "vrd_shape",
}


# ============================================================
# Helpers
# ============================================================
def get_image_path(row):
    if "image_path" in row.index:
        return Path(row["image_path"])
    if "img_path" in row.index:
        return Path(row["img_path"])
    raise ValueError("CSV must contain either 'image_path' or 'img_path'")


def build_prompt(row, task: str) -> str:
    template = PROMPT_TEMPLATES[task]

    if task == "spatial":
        return template.format(
            subj=str(row["subj"]),
            obj=str(row["obj"]),
        )

    elif task == "color":
        return template.format(
            subj=str(row["obj"]),
        )

    elif task == "shape":
        return template.format(
            subj=str(row["obj"]),
        )

    else:
        raise ValueError(f"Unsupported task: {task}")


def get_label(row, task: str):
    if task == "spatial":
        return row["relationship"]
    elif task == "color":
        return row["color"]
    elif task == "shape":
        return row["shape"]
    else:
        raise ValueError(f"Unsupported task: {task}")


def get_sample_id(row, task: str) -> str:
    image_path = str(get_image_path(row))

    if task == "spatial":
        return f"{image_path}||{row['subj']}||{row['obj']}||{row['relationship']}"
    elif task == "color":
        return f"{image_path}||{row['obj']}||{row['color']}"
    elif task == "shape":
        return f"{image_path}||{row['obj']}||{row['shape']}"
    else:
        raise ValueError(f"Unsupported task: {task}")


def save_chunk(chunk_reprs, chunk_labels, chunk_ids, chunk_dir, chunk_idx):
    if not chunk_reprs:
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.npz"

    representations = np.stack(chunk_reprs, axis=0)
    labels = np.array(chunk_labels, dtype=object)
    sample_ids = np.array(chunk_ids, dtype=object)

    np.savez(
        chunk_path,
        representations=representations,
        labels=labels,
        sample_ids=sample_ids,
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
        all_ids.append(data["sample_ids"])

    representations = np.concatenate(all_reprs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sample_ids = np.concatenate(all_ids, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        representations=representations,
        labels=labels,
        sample_ids=sample_ids,
    )

    print(f"\nMerged and saved final file: {output_path}")
    print(f"Final shape: {representations.shape}")
    print(f"{len(labels)} samples saved")


# ============================================================
# Main
# ============================================================
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

    df = pd.read_csv(csv_path)

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
            image = Image.open(image_path).convert("RGB")

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