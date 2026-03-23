# src/extraction/io.py
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_chunk(
    chunk_reprs,
    chunk_labels,
    chunk_ids,
    chunk_dir: Path,
    chunk_idx: int,
) -> None:
    """Save one extraction chunk to disk."""
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


def merge_chunks(chunk_dir: Path, output_path: Path) -> None:
    """Merge chunk_*.npz files into one final representations file."""
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

        if "sample_ids" in data.files:
            all_ids.append(data["sample_ids"])
        elif "image_ids" in data.files:
            all_ids.append(data["image_ids"])
        else:
            raise KeyError(f"No id field in {chunk_file}")

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