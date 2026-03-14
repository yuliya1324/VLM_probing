#!/usr/bin/env python3
"""
make metadata.json for VRD dataset

for synthetic data;
{
  "image_id": "3452448481_a156e6c86c_b",
  "image_filename": "3452448481_a156e6c86c_b.jpg",
  "shape_type": "sign",
  "color_label": "red",
  "prompt": "The color of the sign in the image is"
}

the base csv format:
image_path,obj,color
/Data/masayo.tomita/VLM_probing/data/raw/vrd/sg_train_images/3452448481_a156e6c86c_b.jpg,sign,red
"""



import argparse
import json
from pathlib import Path

import pandas as pd


def build_color_prompt(obj: str) -> str:
    return f"The color of the {obj} in the image is"


def build_shape_prompt(obj: str) -> str:
    return f"The shape of the {obj} in the image is"


def build_spatial_prompt(sub: str, obj: str) -> str:
    return f"The spatial relationship of {sub} to {obj} is"


def normalize_text(x: str) -> str:
    return str(x).strip().lower()


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
        shape = normalize_text(row["shape"])

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
    required = ["img_path", "subj", "obj", "relationship"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for spatial task: {c}")

    metadata = []
    for _, row in df.iterrows():
        image_path = Path(str(row["img_path"]).strip())
        subj = normalize_text(row["subj"])
        obj = normalize_text(row["obj"])
        rel = normalize_text(row["relationship"])

        metadata.append({
            "image_id": image_path.stem,
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "subject_shape": subj,
            "reference_shape": obj,
            "relation": rel,
            "prompt": build_spatial_prompt(subj, obj),
        })
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to VRD CSV")
    parser.add_argument("--task", type=str, required=True, choices=["color", "shape", "spatial"])
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

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

    out_path = out_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {out_path}")
    print(f"N samples: {len(metadata)}")
    if metadata:
        print("Example:")
        print(json.dumps(metadata[0], indent=2))


if __name__ == "__main__":
    main()