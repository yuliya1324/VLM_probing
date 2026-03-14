"""
Make representations_correct.npz from raw_response_predictions.csv.

Keeps only samples where raw VLM response was correct.

Usage:
    python scripts/make_correct_subset.py \
        --pred_csv results/vrd_color/qwen2/raw_response_predictions.csv \
        --repr_npz results/vrd_color/qwen2/representations.npz \
        --out_npz  results/vrd_color/qwen2/representations_correct.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="Path to raw_response_predictions.csv")
    parser.add_argument("--repr_npz", type=str, required=True,
                        help="Path to representations.npz")
    parser.add_argument("--out_npz", type=str, required=True,
                        help="Path to save representations_correct.npz")
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    repr_npz = Path(args.repr_npz)
    out_npz = Path(args.out_npz)

    # 1) load raw-response results and keep only correct rows
    pred_df = pd.read_csv(pred_csv)
    correct_df = pred_df[pred_df["correct"] == True].copy()

    if "sample_id" not in correct_df.columns:
        raise ValueError("pred_csv must contain a 'sample_id' column")

    correct_ids = correct_df["sample_id"].astype(str).tolist()
    correct_id_set = set(correct_ids)

    print(f"Total prediction rows : {len(pred_df)}")
    print(f"Correct rows          : {len(correct_df)}")

    # 2) load representations
    data = np.load(repr_npz, allow_pickle=True)
    representations = data["representations"]
    labels = data["labels"]

    if "sample_ids" in data.files:
        sample_ids = data["sample_ids"]
    elif "image_ids" in data.files:
        sample_ids = data["image_ids"]
    else:
        raise ValueError("repr_npz must contain 'sample_ids' or 'image_ids'")

    sample_ids = np.array(sample_ids).astype(str)

    print(f"Loaded representations: {representations.shape}")
    print(f"Loaded sample_ids      : {len(sample_ids)}")

    # 3) check duplicate IDs in representations
    unique_repr_ids = len(set(sample_ids.tolist()))
    if unique_repr_ids != len(sample_ids):
        raise ValueError(
            "Duplicate sample_ids found in representations. "
            "Please deduplicate first (especially for spatial)."
        )

    # 4) keep only correct sample_ids, preserving representation order
    keep_mask = np.array([sid in correct_id_set for sid in sample_ids], dtype=bool)

    subset_reprs = representations[keep_mask]
    subset_labels = labels[keep_mask]
    subset_ids = sample_ids[keep_mask]

    matched_ids = set(subset_ids.tolist())
    missing_ids = [sid for sid in correct_ids if sid not in matched_ids]

    print(f"Matched correct rows   : {len(subset_ids)}")
    print(f"Missing sample_ids     : {len(missing_ids)}")

    if len(subset_ids) == 0:
        raise ValueError("No matching correct sample_ids found.")

    # 5) save
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        representations=subset_reprs,
        labels=subset_labels,
        sample_ids=subset_ids,
    )

    print(f"\nSaved subset NPZ: {out_npz}")
    print(f"Subset shape    : {subset_reprs.shape}")


if __name__ == "__main__":
    main()