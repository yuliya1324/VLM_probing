import json
import csv
from pathlib import Path
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANN_PATH = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_annotations.json"
IMG_DIR  = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_images"

OUT_CSV  = PROJECT_ROOT / "data" / "vrd_relationships.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d+$")

def _base_label(name: str) -> str:
    """
    'truck'    -> 'truck'
    'truck 2'  -> 'truck'
    'truck 10' -> 'truck'
    """
    name = (name or "").strip()
    m = _SUFFIX_RE.match(name)
    return m.group("base").strip() if m else name

def _is_same_object_family(a: str, b: str) -> bool:
    return _base_label(a) == _base_label(b)

def _load_annotations(ann_path: Path):
    with ann_path.open("r") as f:
        return json.load(f)


def _obj_name(ex, obj_idx: int) -> str:
    objs = ex.get("objects", [])
    if 0 <= obj_idx < len(objs):
        names = objs[obj_idx].get("names", [])
        if names:
            return names[0]
    return f"obj{obj_idx}"


# --------------------------------------------------
# Main CSV creation
# --------------------------------------------------
def build_relationship_csv(rep=None):
    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    if not IMG_DIR.exists():
        raise FileNotFoundError(IMG_DIR)

    data = _load_annotations(ANN_PATH)

    rows = []

    for ex in data:
        img_path = IMG_DIR / ex["filename"]

        # skip missing images
        if not img_path.exists():
            continue

        relationships = ex.get("relationships", [])

        for r in relationships:
            si, oi = r["objects"]

            subj = _obj_name(ex, si)
            obj  = _obj_name(ex, oi)
            rel  = r.get("relationship", "")
            
            # -------------------------------
            # filter out (xxx, xxx N) or (xxx N, xxx M)
            # -------------------------------
            if _is_same_object_family(subj, obj):
                continue
            
            # -------------------------------
            # relationship filtering
            # -------------------------------
            if rep is not None and rel not in rep:
                continue

            rows.append({
                "img_path": str(img_path),
                "subj": subj,
                "obj": obj,
                "relationship": rel
            })

    # --------------------------------------------------
    # write CSV
    # --------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img_path", "subj", "obj", "relationship"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    rep = [
        "right of",
        "left of",
        "below",
        "above",
        #"in front of",
        #"behind",
    ]

    build_relationship_csv(rep=rep)