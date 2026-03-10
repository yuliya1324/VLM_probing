import pandas as pd
import re
from pathlib import Path

# ============================================================
# Paths
# ============================================================
INPUT_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_attributes.csv")
OUT_COLOR_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_color.csv")
OUT_SHAPE_CSV = Path("/Data/masayo.tomita/VLM_probing/data/vrd_shape.csv")


# ============================================================
# Canonical vocabularies
# ============================================================
COLOR_CANONICAL = {
    "black", "white", "gray", "grey", "blue", "red", "green", "yellow",
    "orange", "brown", "pink", "purple", "beige", "tan", "gold", "silver",
    "bronze", "cream", "ivory", "teal", "turquoise", "violet", "lavender",
    "maroon", "navy", "peach", "amber"
}

SHAPE_CANONICAL = {
    "round", "circular", "rectangle", "rectangular", "square",
    "triangle", "triangular", "oval", "octagon", "octagonal",
    "pentagon", "pentagonal", "cylindrical", "sphere", "spherical",
    "cone", "conical", "cube"
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
    "greyish": "grey",
    "bluish": "blue",
    "bluey": "blue",
    "blue-grey": "grey",
    "blue gray": "gray",
    "greenish": "green",
    "reddish": "red",
    "reddish-brown": "brown",
    "yellowish": "yellow",
    "pinkish": "pink",
    "golden": "gold",
    "goldish": "gold",
    "silver\\": "silver",
    "off white": "white",
    "off-white": "white",
    "offwhite": "white",
    "cream colored": "cream",
    "cream-colored": "cream",
    "egg colored": "cream",
    "light blue": "blue",
    "dark blue": "blue",
    "light green": "green",
    "dark green": "green",
    "light brown": "brown",
    "dark brown": "brown",
    "light grey": "grey",
    "dark grey": "grey",
    "light gray": "gray",
    "dark gray": "gray",
    "navy blue": "navy",
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
    "beiege": "beige",
    "biege": "beige",
    "siilver": "silver",
    "sillver": "silver",
    "sivler": "silver",
    "sliver": "silver",
    "torquoise": "turquoise",
    "turqoise": "turquoise",

    # shapes
    "circle": "circular",
    "rectangle": "rectangular",
    "triangle": "triangular",
    "octagon": "octagonal",
    "pentagon": "pentagonal",
    "sphere": "spherical",
    "cone": "conical",
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
    if x in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[x]
    return x


def find_colors(attr: str):
    """
    Return set of canonical colors found in one attribute string.
    Exclude multi-color expressions like 'blue and white', 'red, white, and blue'.
    """
    attr = normalize_attr(attr)

    # explicit multi-color separators -> exclude entirely
    multi_markers = [" and ", ",", "/", "&"]
    if any(m in attr for m in multi_markers):
        matched = set()
        for c in COLOR_CANONICAL:
            if re.search(rf"\b{re.escape(c)}\b", attr):
                matched.add(c)
        if len(matched) >= 2:
            return set()

    found = set()
    for color in COLOR_CANONICAL:
        if re.search(rf"\b{re.escape(color)}\b", attr):
            found.add(color)

    return found


def find_shapes(attr: str):
    """
    Return set of canonical shapes found in one attribute string.
    Exclude attributes containing multiple shapes.
    """
    attr = normalize_attr(attr)

    found = set()
    for shape in SHAPE_CANONICAL:
        if re.search(rf"\b{re.escape(shape)}\b", attr):
            found.add(shape)

    # if a single attribute itself includes multiple shape words, ignore it
    if len(found) >= 2:
        return set()

    return found


# ============================================================
# Main
# ============================================================
def build_color_shape_csv(input_csv, out_color_csv, out_shape_csv):
    df = pd.read_csv(input_csv)

    attr_cols = [c for c in df.columns if c.lower().startswith("attribution")]
    if not attr_cols:
        raise ValueError("No attribution columns found.")

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

            colors_here = find_colors(val)
            shapes_here = find_shapes(val)

            all_colors.update(colors_here)
            all_shapes.update(shapes_here)

        # keep only if exactly one unique color
        if len(all_colors) == 1:
            color_rows.append({
                "image_path": image_path,
                "obj": obj,
                "color": next(iter(all_colors))
            })

        # keep only if exactly one unique shape
        if len(all_shapes) == 1:
            shape_rows.append({
                "image_path": image_path,
                "obj": obj,
                "shape": next(iter(all_shapes))
            })

    color_df = pd.DataFrame(color_rows)
    shape_df = pd.DataFrame(shape_rows)

    color_df.to_csv(out_color_csv, index=False)
    shape_df.to_csv(out_shape_csv, index=False)

    print(f"Saved color CSV: {out_color_csv} ({len(color_df)} rows)")
    print(f"Saved shape CSV: {out_shape_csv} ({len(shape_df)} rows)")


if __name__ == "__main__":
    build_color_shape_csv(INPUT_CSV, OUT_COLOR_CSV, OUT_SHAPE_CSV)