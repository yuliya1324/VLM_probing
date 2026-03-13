"""
Exclude
 - < 1000 area
 - images with same objects
 
Color: downsampling
"""


import pandas as pd
import re
from pathlib import Path

# ============================================================
# Paths
# ============================================================
INPUT_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_csv/vrd_attributes.csv")
OUT_COLOR_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_csv/vrd_color.csv")
OUT_SHAPE_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_csv/vrd_shape.csv")


# ============================================================
# Canonical vocabularies
# ============================================================
COLOR_CANONICAL = {
    "black", "white", "gray", "blue", "red", "green", "yellow",
    "orange", "brown", "pink", "purple",
}

SHAPE_CANONICAL = {
    "round", "circular", "rectangular", "square",
    "triangular", "oval",
}


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
    "circle": "circular",
    "rectangle": "rectangular",
    "triangle": "triangular",
}


# ============================================================
# Helpers
# ============================================================
def normalize_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s\-]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_attr(x: str) -> str:
    x = normalize_text(x)
    return NORMALIZATION_MAP.get(x, x)


def find_colors(attr: str):
    attr = normalize_attr(attr)

    found = set()
    for color in COLOR_CANONICAL:
        if re.search(rf"\b{re.escape(color)}\b", attr):
            found.add(color)

    if len(found) >= 2:
        return set()

    return found


def find_shapes(attr: str):
    attr = normalize_attr(attr)

    found = set()
    for shape in SHAPE_CANONICAL:
        if re.search(rf"\b{re.escape(shape)}\b", attr):
            found.add(shape)

    if len(found) >= 2:
        return set()

    return found


def downsample_by_label(df, label_col, target_per_class=None, random_state=42):
    """
    Downsample each class in label_col.
    - target_per_class=None: use min class size
    - target_per_class=int: keep at most that many per class
    """
    if df.empty:
        return df.copy()

    counts = df[label_col].value_counts().sort_index()

    if target_per_class is None:
        n_target = counts.min()
    else:
        n_target = target_per_class

    sampled = []
    for label, group in df.groupby(label_col, group_keys=False):
        n = min(len(group), n_target)
        sampled.append(group.sample(n=n, random_state=random_state))

    out = pd.concat(sampled, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out


# ============================================================
# Main
# ============================================================
def build_color_shape_csv(
    input_csv,
    out_color_csv,
    out_shape_csv,
    min_area=1000,
    color_target_per_class=None,
    random_state=42,
):
    df = pd.read_csv(input_csv)

    attr_cols = [c for c in df.columns if c.lower().startswith("attribution")]
    if not attr_cols:
        raise ValueError("No attribution columns found.")

    if "area" not in df.columns:
        raise ValueError("Input CSV must contain an 'area' column.")

    # 1. exclude small boxes
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df[df["area"] >= min_area].copy()

    # 2. exclude duplicated (image_path, obj)
    pair_counts = (
        df.groupby(["image_path", "obj"])
        .size()
        .reset_index(name="count")
    )
    valid_pairs = pair_counts[pair_counts["count"] == 1][["image_path", "obj"]]
    df = df.merge(valid_pairs, on=["image_path", "obj"], how="inner")

    # 3. build rows
    color_rows = []
    shape_rows = []

    for _, row in df.iterrows():
        image_path = row["image_path"]
        obj = row["obj"]

        all_colors = set()
        all_shapes = set()

        for col in attr_cols:
            val = row[col]
            if pd.isna(val):
                continue

            val = str(val).strip()
            if not val:
                continue

            all_colors.update(find_colors(val))
            all_shapes.update(find_shapes(val))

        if len(all_colors) == 1:
            color_rows.append({
                "image_path": image_path,
                "obj": obj,
                "color": next(iter(all_colors)),
            })

        if len(all_shapes) == 1:
            shape_rows.append({
                "image_path": image_path,
                "obj": obj,
                "shape": next(iter(all_shapes)),
            })

    color_df = pd.DataFrame(color_rows)
    shape_df = pd.DataFrame(shape_rows)
    
    # deduplicate exact rows
    before_color = len(color_df)
    before_shape = len(shape_df)

    color_df = color_df.drop_duplicates(subset=["image_path", "obj", "color"]).reset_index(drop=True)
    shape_df = shape_df.drop_duplicates(subset=["image_path", "obj", "shape"]).reset_index(drop=True)

    print(f"Removed {before_color - len(color_df)} duplicate color rows")
    print(f"Removed {before_shape - len(shape_df)} duplicate shape rows")

    # 4. downsample only color
    color_downsampled_df = downsample_by_label(
        color_df,
        label_col="color",
        target_per_class=color_target_per_class,
        random_state=random_state,
    )

    # 5. save
    color_df.to_csv(out_color_csv, index=False)
    color_downsampled_df.to_csv(out_color_csv, index=False)
    shape_df.to_csv(out_shape_csv, index=False)

    print(f"Saved color CSV: {out_color_csv} ({len(color_df)} rows)")
    print("Color counts before downsampling:")
    print(color_df["color"].value_counts().sort_index())

    print(f"\nSaved downsampled color CSV: {out_color_csv} ({len(color_downsampled_df)} rows)")
    print("Color counts after downsampling:")
    print(color_downsampled_df["color"].value_counts().sort_index())

    print(f"\nSaved shape CSV: {out_shape_csv} ({len(shape_df)} rows)")


if __name__ == "__main__":
    build_color_shape_csv(
        INPUT_CSV,
        OUT_COLOR_CSV,
        OUT_SHAPE_CSV,
        min_area=1000,
        color_target_per_class=None,   # None -> smallest class size
        random_state=42,
    )

