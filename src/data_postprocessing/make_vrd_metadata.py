#!/usr/bin/env python3
"""
Build VRD metadata JSON from a task-specific CSV.

Usage:
    python scripts/make_vrd_metadata.py --task color
    python scripts/make_vrd_metadata.py --task shape
    python scripts/make_vrd_metadata.py --task spatial

Optional:
    python scripts/make_vrd_metadata.py \
        --task color \
        --csv data/processed/vrd/csv/vrd_color.csv \
        --out_dir data/processed/vrd/metadata
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSVS = {
    "color": PROJECT_ROOT / "data" / "processed" / "vrd" / "csv" / "vrd_color.csv",
    "shape": PROJECT_ROOT / "data" / "processed" / "vrd" / "csv" / "vrd_shape.csv",
    "spatial": PROJECT_ROOT / "data" / "processed" / "vrd" / "csv" / "vrd_spatial.csv",
}

DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "vrd" / "metadata"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build VRD task metadata from a task-specific CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["color", "shape", "spatial"],
        help="Task name.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to VRD CSV. Default: data/processed/vrd/csv/vrd_{task}.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_METADATA_DIR),
        help="Output directory. Default: data/processed/vrd/metadata",
    )

    args = parser.parse_args()

    if args.csv is None:
        args.csv = str(DEFAULT_CSVS[args.task])

    return args


def build_color_prompt(obj: str) -> str:
    return f"The color of the {obj} in the image is"


def build_shape_prompt(obj: str) -> str:
    return f"The shape of the {obj} in the image is"


def build_spatial_prompt(subj: str, obj: str) -> str:
    return f"The spatial relationship of {subj} to {obj} is"


def normalize_text(x: str) -> str:
    return str(x).strip().lower()


def normalize_shape_label(x: str) -> str:
    x = normalize_text(x)
    shape_map = {
        "round": "circular",
    }
    return shape_map.get(x, x)


def normalize_spatial_label(x: str) -> str:
    x = normalize_text(x)
    spatial_map = {
        "left of": "left_of",
        "right of": "right_of",
        "above": "above",
        "below": "below",
    }
    return spatial_map.get(x, x)


def convert_color(df: pd.DataFrame) -> list[dict]:
    required = ["image_path", "obj", "color"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for color task: {c}")

    metadata = []
    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]).strip())
        obj = normalize_text(row["obj"])
        color = normalize_text(row["color"])

        metadata.append({
            "image_id": image_path.stem,
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "shape_type": obj,
            "color_label": color,
            "prompt": build_color_prompt(obj),
        })
    return metadata


def convert_shape(df: pd.DataFrame) -> list[dict]:
    required = ["image_path", "obj", "shape"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for shape task: {c}")

    metadata = []
    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]).strip())
        obj = normalize_text(row["obj"])
        shape = normalize_shape_label(row["shape"])

        metadata.append({
            "image_id": image_path.stem,
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "color_name": obj,
            "shape_label": shape,
            "prompt": build_shape_prompt(obj),
        })
    return metadata


def convert_spatial(df: pd.DataFrame) -> list[dict]:
    required = ["image_path", "subj", "obj", "spatial"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for spatial task: {c}")

    metadata = []
    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]).strip())
        subj = normalize_text(row["subj"])
        obj = normalize_text(row["obj"])
        spatial = normalize_spatial_label(row["spatial"])

        metadata.append({
            "image_id": image_path.stem,
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "subject_shape": subj,
            "reference_shape": obj,
            "relation": spatial,
            "prompt": build_spatial_prompt(subj, obj),
        })
    return metadata


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.task == "color":
        metadata = convert_color(df)
    elif args.task == "shape":
        metadata = convert_shape(df)
    else:
        metadata = convert_spatial(df)

    out_path = out_dir / f"{args.task}_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {out_path}")
    print(f"N samples: {len(metadata)}")
    if metadata:
        print("Example:")
        print(json.dumps(metadata[0], indent=2))


if __name__ == "__main__":
    main()