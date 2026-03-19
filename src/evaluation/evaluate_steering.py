#!/usr/bin/env python3
"""Evaluate steering quality with quantitative metrics.

Computes for each alpha value:
  1. Baseline accuracy (α=0)
  2. Steered-toward-GT accuracy: steer toward correct class, does accuracy improve?
  3. Steered-away accuracy: steer toward wrong class, does accuracy drop?
  4. Target hit rate: when steering toward X, how often does the model say X?
  5. Coherence rate: is the response a valid class label? (vs gibberish)

Usage:
    python scripts/evaluate_steering.py \
        --probes_dir results/qwen2_spatial/probes \
        --data_dir data/raw/spatial \
        --task spatial \
        --model_tag qwen2 \
        --layers 20 \
        --alphas 0 1 2 5 10 20 50 \
        --output results/qwen2_spatial/steering_eval.json \
        --limit 100
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.extraction.extract import MODEL_REGISTRY, build_prompt, get_label
from src.steering.steer import SteeringManager, _generate

# ============================================================
# delete warning message
# ============================================================
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*do_sample.*temperature.*",
)

warnings.filterwarnings(
    "ignore",
    message=r".*do_sample.*top_p.*",
)

warnings.filterwarnings(
    "ignore",
    message=r".*do_sample.*top_k.*",
)

warnings.filterwarnings(
    "ignore",
    message=r".*Plan failed with a cudnnException.*",
)

# ============================================================
# Response parsing
# ============================================================

# Valid class labels per task
VALID_LABELS = {
    "spatial": ["left of", "right of", "above", "below",
                "left_of", "right_of", "left", "right"],
    "color": ["white", "black", "blue", "brown", "green", "red",
              "yellow", "gray", "grey", "orange", "pink", "purple"],
    "shape": ["circular", "oval", "square", "rectangular", "triangular",
              "circle", "rectangle", "triangle", "round", "oblong"],
}

# Normalize model responses to canonical labels
LABEL_NORMALIZE = {
    # spatial
    "left of": "left_of", "left": "left_of", "right of": "right_of",
    "right": "right_of", "above": "above", "below": "below",
    # color
    "grey": "gray",
    # shape
    "circle": "circular", "round": "circular",
    "rectangle": "rectangular", "oblong": "oval",
    "triangle": "triangular",
}


def parse_response(response: str, task: str) -> dict:
    """Parse a model response into structured result."""
    raw = response.strip().lower()
    # Remove punctuation and extra whitespace
    clean = re.sub(r"[^\w\s]", "", raw).strip()

    valid = VALID_LABELS.get(task, [])

    # Check if response contains a valid label
    matched_label = None
    for label in valid:
        if label in clean:
            matched_label = label
            break

    # Normalize
    if matched_label and matched_label in LABEL_NORMALIZE:
        matched_label = LABEL_NORMALIZE[matched_label]

    is_gibberish = len(clean) > 10 and len(set(clean.split())) <= 3

    return {
        "raw": raw,
        "parsed_label": matched_label,
        "is_gibberish": is_gibberish,
    }
    

# ============================================================
# prompt helper
# ============================================================

def get_label_from_metadata(sample: dict, task: str) -> str:
    if task == "color":
        return str(sample["color_label"]).strip().lower()
    elif task == "shape":
        return str(sample["shape_label"]).strip().lower()
    elif task == "spatial":
        label = str(sample["relation"]).strip().lower()
        return LABEL_NORMALIZE.get(label, label)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_prompt_from_metadata(sample: dict, task: str) -> str:
    if "prompt" in sample:
        return sample["prompt"].strip()
    raise KeyError(f"No prompt field found in metadata for task={task}")

def canonicalize_label(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "left of": "left_of",
        "right of": "right_of",
        "left_of": "left_of",
        "right_of": "right_of",
        "above": "above",
        "below": "below",
    }
    return mapping.get(label, label)

def get_probe_classes(probes_dir: str) -> list[str]:
    le = joblib.load(Path(probes_dir) / "label_encoder.joblib")
    return [canonicalize_label(c) for c in le.classes_]

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_steering(
    model, processor, model_tag: str,
    metadata: list, images_dir: str, task: str,
    probes_dir: str, layers: list,
    alphas: list,
    when: str = "all",
    max_new_tokens: int = 50,
) -> dict:
    """Run comprehensive steering evaluation.

    For each sample and each alpha:
      - Steer toward GT: does accuracy improve?
      - Steer toward a random wrong class: does accuracy drop?
    """
    #classes = list(set(get_label(s, task) for s in metadata))
    classes = list(set(get_label_from_metadata(s, task) for s in metadata))
    n = len(metadata)

    # Results per alpha
    results_by_alpha = {}

    for alpha in alphas:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")
        print(f"{'='*50}")

        stats = {
            "baseline_correct": 0,
            "steer_toward_gt_correct": 0,
            "steer_away_correct": 0,       # still correct despite wrong steering
            "steer_away_hit_target": 0,     # response matches the wrong steered class
            # "coherent": 0,
            "gibberish": 0,
            "total": 0,
            "per_class_hit": defaultdict(int),
            "per_class_total": defaultdict(int),
        }

        for i, sample in enumerate(metadata):
            if "image_path" in sample:
                image_path = Path(sample["image_path"])
            else:
                image_path = Path(images_dir) / sample["image_filename"]
            image = Image.open(image_path).convert("RGB")
            #prompt = build_prompt(sample, task)
            #gt_label = get_label(sample, task)
            prompt = get_prompt_from_metadata(sample, task)
            gt_label = get_label_from_metadata(sample, task)

            # Normalize GT
            gt_norm = LABEL_NORMALIZE.get(gt_label, gt_label)

            # Pick a wrong class for "steer away" experiment
            wrong_classes = [c for c in classes if c != gt_label]
            if not wrong_classes:
                continue
            wrong_class = wrong_classes[i % len(wrong_classes)]

            try:
                # --- Baseline (always computed, even at α>0 for comparison) ---
                baseline_resp = _generate(model, processor, model_tag, image, prompt, max_new_tokens)
                baseline_parsed = parse_response(baseline_resp, task)
                baseline_correct = baseline_parsed["parsed_label"] == gt_norm

                if alpha == 0:
                    # No steering, just baseline
                    stats["baseline_correct"] += int(baseline_correct)
                    stats["steer_toward_gt_correct"] += int(baseline_correct)
                    stats["steer_away_correct"] += int(baseline_correct)
                    # stats["coherent"] += int(baseline_parsed["is_coherent"])
                    stats["gibberish"] += int(baseline_parsed["is_gibberish"])
                else:
                    # --- Steer toward GT ---
                    with SteeringManager.from_probes(
                        model, model_tag, probes_dir, layers,
                        target_class=gt_label, alpha=alpha,
                        strategy="push", when=when,
                    ):
                        toward_resp = _generate(model, processor, model_tag, image, prompt, max_new_tokens)
                    toward_parsed = parse_response(toward_resp, task)
                    stats["steer_toward_gt_correct"] += int(toward_parsed["parsed_label"] == gt_norm)

                    # --- Steer away from GT (toward wrong class) ---
                    with SteeringManager.from_probes(
                        model, model_tag, probes_dir, layers,
                        target_class=wrong_class, alpha=alpha,
                        strategy="push", when=when,
                    ):
                        away_resp = _generate(model, processor, model_tag, image, prompt, max_new_tokens)
                    away_parsed = parse_response(away_resp, task)
                    wrong_norm = LABEL_NORMALIZE.get(wrong_class, wrong_class)
                    stats["steer_away_correct"] += int(away_parsed["parsed_label"] == gt_norm)
                    stats["steer_away_hit_target"] += int(away_parsed["parsed_label"] == wrong_norm)
                    # stats["coherent"] += int(away_parsed["is_coherent"])
                    stats["gibberish"] += int(away_parsed["is_gibberish"])

                    stats["baseline_correct"] += int(baseline_correct)

                    # Per-class tracking for target hit rate
                    stats["per_class_total"][wrong_class] += 1
                    if away_parsed["parsed_label"] == wrong_norm:
                        stats["per_class_hit"][wrong_class] += 1

                stats["total"] += 1

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{n}] processed")

            except Exception as e:
                print(f"  [{i+1}/{n}] error: {e}")

        total = max(stats["total"], 1)
        results_by_alpha[alpha] = {
            "alpha": alpha,
            "total": stats["total"],
            "baseline_accuracy": stats["baseline_correct"] / total,
            "steer_toward_gt_accuracy": stats["steer_toward_gt_correct"] / total,
            "steer_away_accuracy": stats["steer_away_correct"] / total,
            "steer_away_target_hit_rate": stats["steer_away_hit_target"] / total,
            # "coherence_rate": stats["coherent"] / total,
            "gibberish_rate": stats["gibberish"] / total,
            "per_class_hit_rate": {
                cls: stats["per_class_hit"][cls] / max(stats["per_class_total"][cls], 1)
                for cls in stats["per_class_total"]
            },
        }

        r = results_by_alpha[alpha]
        print(f"  Baseline acc:        {r['baseline_accuracy']:.3f}")
        print(f"  Steer→GT acc:        {r['steer_toward_gt_accuracy']:.3f}")
        print(f"  Steer→wrong acc:     {r['steer_away_accuracy']:.3f}")
        print(f"  Steer→wrong hit:     {r['steer_away_target_hit_rate']:.3f}")
        # print(f"  Coherence:           {r['coherence_rate']:.3f}")
        print(f"  Gibberish:           {r['gibberish_rate']:.3f}")

    return {
        "task": task,
        "layers": layers,
        "when": when,
        "classes": classes,
        "results_by_alpha": results_by_alpha,
    }


# ============================================================
# Plotting
# ============================================================

def plot_steering_eval(results: dict, output_path: str = None):
    """Plot steering metrics vs alpha."""
    import matplotlib.pyplot as plt

    alphas = []
    baseline = []
    toward_gt = []
    away_acc = []
    away_hit = []
    gibberish = []

    for alpha, r in sorted(results["results_by_alpha"].items(), key=lambda x: x[0]):
        alphas.append(alpha)
        baseline.append(r["baseline_accuracy"])
        toward_gt.append(r["steer_toward_gt_accuracy"])
        away_acc.append(r["steer_away_accuracy"])
        away_hit.append(r["steer_away_target_hit_rate"])
        gibberish.append(r["gibberish_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot(alphas, baseline, "k--", marker="o", markersize=4, label="Baseline", alpha=0.5)
    ax.plot(alphas, toward_gt, "g-", marker="s", markersize=4, label="Steer → GT")
    ax.plot(alphas, away_acc, "r-", marker="^", markersize=4, label="Steer → wrong (still correct)")
    ax.plot(alphas, away_hit, "m-", marker="D", markersize=4, label="Steer → wrong (target hit rate)")
    ax.set_xlabel("α")
    ax.set_ylabel("Accuracy")
    ax.set_title("Steering Effect on Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # # Panel 2: Target hit rate
    # ax = axes[1]
    # ax.plot(alphas, away_hit, "m-", marker="D", markersize=4, label="Steer → wrong: target hit rate")
    # ax.set_xlabel("α")
    # ax.set_ylabel("Hit Rate")
    # ax.set_title("Does Steering Change the Response?")
    # ax.legend()
    # ax.grid(alpha=0.3)
    # ax.set_ylim(-0.05, 1.05)

    # Panel 3: Coherence / gibberish
    ax = axes[1]
    ax.plot(alphas, gibberish, "r--", marker="x", markersize=4, label="Gibberish")
    ax.set_xlabel("α")
    ax.set_ylabel("Rate")
    ax.set_title("Response Quality vs α")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot → {output_path}")
    else:
        plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate steering quality")
    parser.add_argument("--probes_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["spatial", "color", "shape"])
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument("--when", type=str, default="all", choices=["all", "prefill"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    parser.add_argument("--plot", type=str, default=None, help="Save plot")
    args = parser.parse_args()

    # Load metadata
    data_dir = Path(args.data_dir)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    if args.limit:
        metadata = metadata[:args.limit]

    # Load model
    registry_entry = MODEL_REGISTRY[args.model_tag]
    model_id = args.model_id or registry_entry["default_id"]
    print(f"Loading model: {model_id}")
    model, processor = registry_entry["loader"](model_id)

    # Run evaluation
    results = evaluate_steering(
        model, processor, args.model_tag,
        metadata, str(data_dir / "images"), args.task,
        args.probes_dir, args.layers,
        alphas=args.alphas,
        when=args.when,
        max_new_tokens=args.max_new_tokens,
    )

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved results → {args.output}")

    # Plot
    if args.plot:
        plot_steering_eval(results, args.plot)


if __name__ == "__main__":
    main()