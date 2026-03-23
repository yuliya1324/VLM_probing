#src/data_preprocessing/build_task_csv.py
"""
Build task-specific VRD CSVs.

Usage:
    python src/data_preprocessing/build_task_csv.py --task color
    python src/data_preprocessing/build_task_csv.py --task shape
    python src/data_preprocessing/build_task_csv.py --task spatial

Tasks:
- color   : build balanced color CSV from vrd_base.csv
- shape   : build shape CSV from vrd_base.csv
- spatial : build balanced spatial CSV from raw VRD annotations

Outputs:
- data/processed/vrd/csv/vrd_{task}.csv
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from common import ANN_PATH, IMG_DIR, VRD_CSV_DIR, load_annotations, base_label


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CSV = VRD_CSV_DIR / "vrd_base.csv"

OUT_COLOR_CSV = VRD_CSV_DIR / "vrd_color.csv"
OUT_SHAPE_CSV = VRD_CSV_DIR / "vrd_shape.csv"
OUT_SPATIAL_CSV = VRD_CSV_DIR / "vrd_spatial.csv"

# ============================================================
# Canonical vocabularies
# ============================================================
COLOR_CANONICAL = {
    "black", "white", "gray", "blue", "red", "green", "yellow",
    "orange", "brown", "pink", "purple",
}

# Final classes are 5. "round" is accepted in raw data but normalized to "circular".
SHAPE_CANONICAL = {
    "circular", "rectangular", "square", "triangular", "oval",
}

DEFAULT_SPATIAL_LABELS = ["right_of", "left_of", "below", "above"]


# ============================================================
# typo / variant normalization
# ============================================================
NORMALIZATION_MAP = {
    # colors
    "gry": "gray",
    "gra y": "gray",
    "grayish": "gray",
    "graying": "gray",
    "grey": "gray",
    "greyish": "gray",
    "blue-grey": "gray",
    "blue gray": "gray",
    "bluish": "blue",
    "bluey": "blue",
    "greenish": "green",
    "reddish": "red",
    "reddish-brown": "brown",
    "yellowish": "yellow",
    "pinkish": "pink",
    "off white": "white",
    "off-white": "white",
    "offwhite": "white",
    "light blue": "blue",
    "dark blue": "blue",
    "light green": "green",
    "dark green": "green",
    "light brown": "brown",
    "dark brown": "brown",
    "light grey": "gray",
    "dark grey": "gray",
    "light gray": "gray",
    "dark gray": "gray",
    "lime green": "green",
    "olive green": "green",
    "bright green": "green",

    # common typos
    "blac": "black",
    "blacck": "black",
    "blakc": "black",
    "b;ack": "black",
    "blacky": "black",
    "whie": "white",
    "whiet": "white",
    "whire": "white",
    "whte": "white",
    "whtie": "white",
    "yello": "yellow",
    "yelow": "yellow",
    "oragne": "orange",
    "orangle": "orange",
    "ornage": "orange",
    "gren": "green",
    "gree": "green",
    "geren": "green",
    "brow": "brown",
    "bronw": "brown",
    "brone": "brown",

    # shapes
    "round": "circular",
    "circle": "circular",
    "rectangle": "rectangular",
    "triangle": "triangular",
}

SPATIAL_NORMALIZATION_MAP = {
    "left of": "left_of",
    "right of": "right_of",
    "above": "above",
    "below": "below",
}
# ============================================================
# Shared helpers
# ============================================================
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d+$")


def normalize_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s\-]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_attr(x: str) -> str:
    x = normalize_text(x)
    return NORMALIZATION_MAP.get(x, x)


def normalize_spatial_label(x: str) -> str:
    x = normalize_text(x)
    return SPATIAL_NORMALIZATION_MAP.get(x, x)


def find_labels(attr: str, canonical_set: set[str]) -> set[str]:
    """
    Return matching labels from canonical_set.
    If >=2 labels are found in one attribute string, return empty set.
    """
    attr = normalize_attr(attr)

    found = set()
    for label in canonical_set:
        if re.search(rf"\b{re.escape(label)}\b", attr):
            found.add(label)

    if len(found) >= 2:
        return set()

    return found


def downsample_df_by_label(
    df: pd.DataFrame,
    label_col: str,
    target_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    counts = df[label_col].value_counts().sort_index()
    n_target = counts.min() if target_per_class is None else target_per_class

    sampled = []
    for label, group in df.groupby(label_col, group_keys=False):
        n = min(len(group), n_target)
        sampled.append(group.sample(n=n, random_state=random_state))

    return (
        pd.concat(sampled, axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def downsample_rows_by_label(
    rows: list[dict],
    label_col: str,
    seed: int = 42,
    max_per_class: int | None = None,
) -> list[dict]:
    if not rows:
        return rows

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[label_col]].append(row)

    if max_per_class is None:
        target_n = min(len(items) for items in grouped.values())
    else:
        target_n = min(max_per_class, min(len(items) for items in grouped.values()))

    balanced_rows = []
    for label, items in grouped.items():
        balanced_rows.extend(rng.sample(items, target_n))

    rng.shuffle(balanced_rows)

    grouped_after = defaultdict(list)
    for row in balanced_rows:
        grouped_after[row[label_col]].append(row)

    return balanced_rows

def obj_name_from_index(ex: dict, obj_idx: int) -> str:
    objs = ex.get("objects", [])
    if 0 <= obj_idx < len(objs):
        names = objs[obj_idx].get("names", [])
        if names:
            return names[0]
    return f"obj{obj_idx}"


def is_same_object_family(a: str, b: str) -> bool:
    return base_label(a) == base_label(b)


# ============================================================
# Color / Shape builders
# ============================================================
def _load_base_df(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"{input_csv} not found. Run your base CSV builder first."
        )

    df = pd.read_csv(input_csv)

    attr_cols = [c for c in df.columns if c.lower().startswith("attribution")]
    if not attr_cols:
        raise ValueError("No attribution columns found.")
    if "area" not in df.columns:
        raise ValueError("Input CSV must contain an 'area' column.")
    if "image_path" not in df.columns or "obj" not in df.columns:
        raise ValueError("Input CSV must contain 'image_path' and 'obj' columns.")

    return df


def _filter_object_rows(df: pd.DataFrame, min_area: int) -> tuple[pd.DataFrame, list[str]]:
    attr_cols = [c for c in df.columns if c.lower().startswith("attribution")]

    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df[df["area"] >= min_area].copy()

    # Exclude duplicated (image_path, obj)
    pair_counts = (
        df.groupby(["image_path", "obj"])
        .size()
        .reset_index(name="count")
    )
    valid_pairs = pair_counts[pair_counts["count"] == 1][["image_path", "obj"]]
    df = df.merge(valid_pairs, on=["image_path", "obj"], how="inner")

    return df, attr_cols


def build_color_csv(
    input_csv: Path = BASE_CSV,
    output_csv: Path = OUT_COLOR_CSV,
    min_area: int = 1000,
    target_per_class: int | None = None,
    random_state: int = 42,
) -> None:
    df = _load_base_df(input_csv)
    df, attr_cols = _filter_object_rows(df, min_area=min_area)

    rows = []
    for _, row in df.iterrows():
        all_colors = set()

        for col in attr_cols:
            val = row[col]
            if pd.isna(val):
                continue
            val = str(val).strip()
            if not val:
                continue

            all_colors.update(find_labels(val, COLOR_CANONICAL))

        if len(all_colors) == 1:
            rows.append({
                "image_path": row["image_path"],
                "obj": row["obj"],
                "color": next(iter(all_colors)),
            })

    color_df = pd.DataFrame(rows).drop_duplicates(
        subset=["image_path", "obj", "color"]
    ).reset_index(drop=True)

    color_df = downsample_df_by_label(
        color_df,
        label_col="color",
        target_per_class=target_per_class,
        random_state=random_state,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    color_df.to_csv(output_csv, index=False)

    print(f"\nSaved color CSV: {output_csv} ({len(color_df)} rows)")
    print("Color counts:")
    if not color_df.empty:
        print(color_df["color"].value_counts().sort_index())


def build_shape_csv(
    input_csv: Path = BASE_CSV,
    output_csv: Path = OUT_SHAPE_CSV,
    min_area: int = 1000,
) -> None:
    df = _load_base_df(input_csv)
    df, attr_cols = _filter_object_rows(df, min_area=min_area)

    rows = []
    for _, row in df.iterrows():
        all_shapes = set()

        for col in attr_cols:
            val = row[col]
            if pd.isna(val):
                continue
            val = str(val).strip()
            if not val:
                continue

            all_shapes.update(find_labels(val, SHAPE_CANONICAL))

        if len(all_shapes) == 1:
            rows.append({
                "image_path": row["image_path"],
                "obj": row["obj"],
                "shape": next(iter(all_shapes)),
            })

    shape_df = pd.DataFrame(rows).drop_duplicates(
        subset=["image_path", "obj", "shape"]
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shape_df.to_csv(output_csv, index=False)

    print(f"Saved shape CSV: {output_csv} ({len(shape_df)} rows)")
    print("Shape counts:")
    if not shape_df.empty:
        print(shape_df["shape"].value_counts().sort_index())


# ============================================================
# Spatial builder
# ============================================================
def _deduplicate_spatial_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row["image_path"],
            row["subj"],
            row["obj"],
            row["spatial"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def build_spatial_csv(
    output_csv: Path = OUT_SPATIAL_CSV,
    spatial_labels: list[str] | None = None,
    downsample: bool = True,
    max_per_class: int | None = None,
    seed: int = 42,
) -> None:
    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    if not IMG_DIR.exists():
        raise FileNotFoundError(IMG_DIR)

    data = load_annotations()
    rows = []

    for ex in data:
        filename = ex.get("filename")
        if filename is None:
            continue

        image_path = IMG_DIR / filename
        if not image_path.exists():
            continue

        for rel_ex in ex.get("relationships", []):
            si, oi = rel_ex["objects"]

            subj = obj_name_from_index(ex, si)
            obj = obj_name_from_index(ex, oi)
            spatial_raw = rel_ex.get("relationship", "")
            spatial = normalize_spatial_label(spatial_raw)

            if is_same_object_family(subj, obj):
                continue

            if spatial_labels is not None and spatial not in spatial_labels:
                continue

            rows.append({
                "image_path": str(image_path),
                "subj": subj,
                "obj": obj,
                "spatial": spatial,
            })

    rows = _deduplicate_spatial_rows(rows)

    if downsample:
        rows = downsample_rows_by_label(
            rows,
            label_col="spatial",
            seed=seed,
            max_per_class=max_per_class,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    spatial_df = pd.DataFrame(rows, columns=["image_path", "subj", "obj", "spatial"])
    spatial_df.to_csv(output_csv, index=False)

    print(f"\nSaved spatial CSV: {output_csv} ({len(spatial_df)} rows)")
    print("Spatial counts:")
    if not spatial_df.empty:
        print(spatial_df["spatial"].value_counts().sort_index())


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["color", "shape", "spatial"],
        help="Task to build.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum bbox area for color/shape tasks.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Target per class for balanced color/spatial datasets. Default: minimum class size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing.",
    )
    parser.add_argument(
        "--spatial-labels",
        nargs="*",
        default=DEFAULT_SPATIAL_LABELS,
        help="Spatial labels to keep for spatial task.",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Disable downsampling for spatial task.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "color":
        build_color_csv(
            input_csv=BASE_CSV,
            output_csv=OUT_COLOR_CSV,
            min_area=args.min_area,
            target_per_class=args.target_per_class,
            random_state=args.seed,
        )

    elif args.task == "shape":
        build_shape_csv(
            input_csv=BASE_CSV,
            output_csv=OUT_SHAPE_CSV,
            min_area=args.min_area,
        )

    elif args.task == "spatial":
        build_spatial_csv(
            output_csv=OUT_SPATIAL_CSV,
            spatial_labels=args.spatial_labels,
            downsample=not args.no_downsample,
            max_per_class=args.target_per_class,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()