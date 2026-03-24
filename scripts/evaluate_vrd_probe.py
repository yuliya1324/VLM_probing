# scripts/evaluate_vrd_probe.py
"""
Evaluate probe accuracy on VRD representations and save per-sample predictions.

Usage:
    python scripts/evaluate_vrd_probe.py \
        --task spatial \
        --model_tag qwen2 \
        --probes_dir results/qwen2_spatial/probes

    python scripts/evaluate_vrd_probe.py \
        --task color \
        --model_tag llava15 \
        --probes_dir results/llava15_color/probes \
        --layer 24
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing.probe import load_probe, load_best_probe, predict_with_probe
from src.data_preprocessing.vrd import (
    TASK_TO_CSV,
    load_vrd_dataframe,
    get_image_path,
    build_prompt,
    get_label,
    get_sample_id,
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

TASK_TO_REPR = {
    "spatial": lambda model_tag: RESULTS_DIR / "vrd" / "spatial" / model_tag / "representations.npz",
    "color": lambda model_tag: RESULTS_DIR / "vrd" / "color" / model_tag / "representations.npz",
    "shape": lambda model_tag: RESULTS_DIR / "vrd" / "shape" / model_tag / "representations.npz",
}


def build_metadata_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = out.apply(lambda row: get_sample_id(row, task), axis=1)
    out["image_path_eval"] = out.apply(lambda row: str(get_image_path(row)), axis=1)
    out["prompt"] = out.apply(lambda row: build_prompt(row, task), axis=1)
    out["ground_truth_raw"] = out.apply(lambda row: str(get_label(row, task)), axis=1)
    return out

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["spatial", "color", "shape"])
    parser.add_argument("--model_tag", type=str, required=True, choices=["qwen2", "llava15", "vila"])
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--repr_path", type=str, default=None)
    parser.add_argument("--probes_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else TASK_TO_CSV[args.task]
    repr_path = Path(args.repr_path) if args.repr_path else TASK_TO_REPR[args.task](args.model_tag)

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
    else:
        output_csv = repr_path.parent / "probe_predictions.csv"

    probes_dir = Path(args.probes_dir)

    print(f"Task       : {args.task}")
    print(f"Model tag  : {args.model_tag}")
    print(f"CSV path   : {csv_path}")
    print(f"Repr path  : {repr_path}")
    print(f"Probes dir : {probes_dir}")
    print(f"Output CSV : {output_csv}")

    # ----------------------------
    # Load metadata CSV
    # ----------------------------
    df = load_vrd_dataframe(args.task, str(csv_path))
    df = build_metadata_df(df, args.task)

    if df["sample_id"].duplicated().any():
        dup_count = int(df["sample_id"].duplicated().sum())
        print(f"Warning: found {dup_count} duplicated sample_id rows in CSV")

    meta_by_id = df.set_index("sample_id").to_dict(orient="index")

    # ----------------------------
    # Load representations
    # ----------------------------
    data = np.load(repr_path, allow_pickle=True)
    representations = data["representations"]   # (N, n_layers, hidden_dim)
    sample_ids = data["sample_ids"] if "sample_ids" in data.files else data["image_ids"]

    print(f"Loaded representations: {representations.shape}")

    # ----------------------------
    # Load probe
    # ----------------------------
    if args.layer is None:
        probe, le, layer_idx = load_best_probe(str(probes_dir))
        print(f"Loaded best probe: layer {layer_idx}")
    else:
        probe, le = load_probe(str(probes_dir), args.layer)
        layer_idx = args.layer
        print(f"Loaded probe: layer {layer_idx}")

    probe_classes = [str(x) for x in le.classes_]
    print(f"Probe classes: {sorted(probe_classes)}")

    # ----------------------------
    # Predict
    # ----------------------------
    rows = []
    missing_meta = 0

    for i in tqdm(range(len(sample_ids)), desc="Evaluating"):
        sample_id = str(sample_ids[i])

        if sample_id not in meta_by_id:
            missing_meta += 1
            continue

        meta = meta_by_id[sample_id]
        hidden_state = representations[i, layer_idx]   # (hidden_dim,)
        result = predict_with_probe(probe, le, hidden_state)

        gt_raw = str(meta["ground_truth_raw"])
        pred_raw = str(result["prediction"])

        gt_eval = gt_raw.strip().lower()
        pred_eval = pred_raw.strip().lower()
        correct = pred_eval == gt_eval
        probs = result.get("probabilities", {})
        confidence = max(probs.values()) if len(probs) > 0 else None

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": meta["image_path_eval"],
                "prompt": meta["prompt"],
                "ground_truth": gt_raw,
                "ground_truth_eval": gt_eval,
                "answer": pred_raw,
                "answer_eval": pred_eval,
                "correct": bool(correct),
                "confidence": confidence,
                "layer": layer_idx,
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    # ----------------------------
    # Summary
    # ----------------------------
    evaluated = len(out_df)
    total_csv = len(df)
    total_repr = len(sample_ids)
    num_correct = int(out_df["correct"].sum()) if evaluated > 0 else 0
    accuracy = float(out_df["correct"].mean()) if evaluated > 0 else 0.0

    print("\n===== Summary =====")
    print(f"CSV rows             : {total_csv}")
    print(f"Representation rows  : {total_repr}")
    print(f"Evaluated rows       : {evaluated}")
    print(f"Missing metadata     : {missing_meta}")
    print(f"Correct              : {num_correct}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Saved CSV            : {output_csv}")

    if evaluated > 0:
        print("\nPer-class accuracy:")
        print(
            out_df.groupby("ground_truth_eval")["correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .sort_values("count", ascending=False)
        )


if __name__ == "__main__":
    main()