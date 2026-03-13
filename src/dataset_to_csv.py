import json
import csv
from pathlib import Path
import re
import random
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANN_PATH = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_annotations.json"
IMG_DIR  = PROJECT_ROOT / "data" / "raw" / "vrd" / "sg_train_images"

OUT_CSV  = PROJECT_ROOT / "data" / "vrd_csv" / "vrd_spatial.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d+$")

def _base_label(name: str) -> str:
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

def _downsample_by_relationship(rows, seed=42, max_per_class=None):
    rng = random.Random(seed)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["relationship"]].append(row)

    print("Before downsampling:")
    for rel, items in grouped.items():
        print(f"  {rel}: {len(items)}")

    if max_per_class is None:
        target_n = min(len(items) for items in grouped.values())
    else:
        target_n = min(
            max_per_class,
            min(len(items) for items in grouped.values())
        )

    balanced_rows = []
    for rel, items in grouped.items():
        sampled = rng.sample(items, target_n)
        balanced_rows.extend(sampled)

    rng.shuffle(balanced_rows)

    grouped_after = defaultdict(list)
    for row in balanced_rows:
        grouped_after[row["relationship"]].append(row)

    print(f"\nAfter downsampling (target per class = {target_n}):")
    for rel, items in grouped_after.items():
        print(f"  {rel}: {len(items)}")

    return balanced_rows

def _deduplicate_rows(rows):
    seen = set()
    deduped = []
    dup_count = 0

    for row in rows:
        key = (
            row["img_path"],
            row["subj"],
            row["obj"],
            row["relationship"],
        )
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(row)

    print(f"Removed {dup_count} duplicated rows")
    return deduped

# --------------------------------------------------
# Main CSV creation
# --------------------------------------------------
def build_relationship_csv(rep=None, downsample=False, max_per_class=None, seed=42):
    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    if not IMG_DIR.exists():
        raise FileNotFoundError(IMG_DIR)

    data = _load_annotations(ANN_PATH)
    rows = []

    for ex in data:
        img_path = IMG_DIR / ex["filename"]

        if not img_path.exists():
            continue

        relationships = ex.get("relationships", [])

        for r in relationships:
            si, oi = r["objects"]

            subj = _obj_name(ex, si)
            obj  = _obj_name(ex, oi)
            rel  = r.get("relationship", "")

            if _is_same_object_family(subj, obj):
                continue

            if rep is not None and rel not in rep:
                continue

            rows.append({
                "img_path": str(img_path),
                "subj": subj,
                "obj": obj,
                "relationship": rel
            })
            
    print(f"Before deduplication: {len(rows)} rows")
    rows = _deduplicate_rows(rows)
    print(f"After deduplication: {len(rows)} rows")

    if downsample:
        rows = _downsample_by_relationship(
            rows,
            seed=seed,
            max_per_class=max_per_class
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img_path", "subj", "obj", "relationship"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    rep = [
        "right of",
        "left of",
        "below",
        "above",
    ]

    build_relationship_csv(
        rep=rep,
        downsample=True,
        max_per_class=None,   # None -> downsample to classes with minimum
        seed=42
    )